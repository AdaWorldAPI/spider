//! `spider_doc_ir` — the **DOM retina** (W3 of the OGAR doc-IR × spider
//! convergence, `AdaWorldAPI/OGAR docs/DOC-IR-SPIDER-CONVERGENCE-PLAN.md`).
//!
//! Harvests an HTML page into the source-agnostic [`ogar_doc_ir`] perceptual
//! IR, using the same streaming `lol_html` parser `spider_agent_html` already
//! uses. A web page and a scanned document thus produce ONE shape — the whole
//! point of the convergence: everything upstream (`ogar-doc` persistence,
//! DeepNSM, the discovery arm, MedCare-rs as an abstract document store)
//! consumes the same region tree regardless of retina.
//!
//! # Why lol_html fits
//!
//! HTML5 sectioning is *self-labelling*: `<header>/<main>/<footer>/<table>`
//! are the Kopfzeile/main/Fußzeile positions OCR must *infer*. Each is a CSS
//! selector → a `lol_html` element handler. The rewriter is forward-only
//! streaming, so:
//!
//! - **reading order is free** — handler fire order = document order = the
//!   temporal stream DeepNSM consumes;
//! - **the v1 spatial rail is free** — the stream index quantizes top-to-
//!   bottom onto the unit-square [`Rail`], the honest `Geometry::DomOrder`
//!   pseudo-geometry (rendered `getBoundingClientRect` rails are a later
//!   increment behind the `chrome` feature; the provenance lane records which
//!   mode produced the rails).
//!
//! # Source-agnostic by construction
//!
//! [`harvest`] emits only closed-vocabulary [`RegionKind`]s, and its output
//! passes the SAME [`ogar_doc_ir::from_json`] load gate an OCR producer's
//! output does (asserted in the tests). The IR never learns which retina made
//! it beyond the [`Provenance`] lane.

use std::cell::RefCell;
use std::rc::Rc;

use lol_html::{element, text, HtmlRewriter, Settings};
use ogar_doc_ir::{
    BBoxRail, DocIr, DocPage, Geometry, Provenance, Rail, Region, RegionKind, TableCell,
    TypedField, DOC_IR_VERSION,
};
use sha2::{Digest, Sha256};

/// The HTML5 landmark selectors we lift to regions, plus the table-cell
/// selectors. Kept as one string so the element/text handlers share it.
const LANDMARKS: &str = "header, footer, main, article, nav, table, figure, img";
const CELLS: &str = "td, th";
/// Structured-data carriers we lift to [`TypedField`]s: `<meta>` tags whose
/// `content` attribute carries a value keyed by `name` / `property`
/// (OpenGraph, Dublin Core) / `itemprop` (microdata). Inline microdata
/// (`[itemprop]` with text) and JSON-LD (`<script type=ld+json>`) are named
/// later increments — the DOM analogue of OCR's `harvest_profile` grows
/// carrier by carrier without changing the IR.
const META_FIELDS: &str = "meta[name][content], meta[property][content], meta[itemprop][content]";

/// A DOM-declared field carries higher trust than a recognized one — this is
/// the DOM end of the [`Provenance`]-scoped confidence lane (a value the page
/// author literally typed, not one an OCR pass guessed).
const DOM_DECLARED_CONFIDENCE: u8 = 255;

/// Map an HTML5 landmark tag to the closed [`RegionKind`] vocabulary.
fn kind_of(tag: &str) -> RegionKind {
    match tag {
        "header" => RegionKind::Header,
        "footer" => RegionKind::Footer,
        "main" | "article" => RegionKind::Main,
        "nav" => RegionKind::Nav,
        "table" => RegionKind::Table,
        "figure" | "img" => RegionKind::Figure,
        _ => RegionKind::Text,
    }
}

/// A DomOrder rail band for the region at stream index `order` — a coarse
/// top-to-bottom placement on the 256-tall unit tile. Rendered geometry (the
/// `chrome` `bounding_box()` path) replaces this in a later increment.
fn dom_order_bbox(order: u16) -> BBoxRail {
    let y = order.min(255) as u8;
    BBoxRail {
        tl: Rail { x: 0, y },
        br: Rail {
            x: 255,
            y: order.saturating_add(1).min(255) as u8,
        },
    }
}

/// A region accumulating as the stream flows.
struct Pending {
    kind: RegionKind,
    order: u16,
    text: String,
    cells: Vec<TableCell>,
}

/// Harvest one HTML page into a [`DocIr`]. `content` is the raw HTML bytes as
/// received (the [`DocIr::content_sha256`] is taken over them — the
/// cross-retina convergence key).
///
/// v1 scope: top-level HTML5 landmarks → regions in reading order; `<td>`/
/// `<th>` under a table → [`TableCell`]s on the open table region; landmark
/// text captured; `<meta name|property|itemprop … content>` → document-level
/// [`TypedField`]s. Rendered geometry, inline microdata, and JSON-LD are
/// named later increments (see the module doc / the convergence plan).
#[must_use]
pub fn harvest(content: &str) -> DocIr {
    let pending: Rc<RefCell<Vec<Pending>>> = Rc::new(RefCell::new(Vec::new()));
    let counter = Rc::new(RefCell::new(0u16));
    let cell_counter = Rc::new(RefCell::new(0u16));
    let fields: Rc<RefCell<Vec<TypedField>>> = Rc::new(RefCell::new(Vec::new()));

    let (p_el, c_el) = (pending.clone(), counter.clone());
    let p_tx = pending.clone();
    let (p_cell, cc) = (pending.clone(), cell_counter.clone());
    let f_meta = fields.clone();

    let mut sink = Vec::new();
    {
        let mut rw = HtmlRewriter::new(
            Settings {
                element_content_handlers: vec![
                    // Open a region on each landmark start tag, in stream order.
                    element!(LANDMARKS, move |el| {
                        let mut order = c_el.borrow_mut();
                        p_el.borrow_mut().push(Pending {
                            kind: kind_of(el.tag_name().as_str()),
                            order: *order,
                            text: String::new(),
                            cells: Vec::new(),
                        });
                        *order += 1;
                        // A new table resets the per-table cell running index.
                        Ok(())
                    }),
                    // Append a cell to the currently-open table region.
                    text!(CELLS, move |t| {
                        let mut regions = p_cell.borrow_mut();
                        if let Some(tbl) = regions
                            .iter_mut()
                            .rev()
                            .find(|p| p.kind == RegionKind::Table)
                        {
                            let s = t.as_str();
                            if !s.trim().is_empty() {
                                let mut n = cc.borrow_mut();
                                tbl.cells.push(TableCell {
                                    row: (*n / 16) as u8,
                                    col: (*n % 16) as u8,
                                    text: s.trim().to_string(),
                                    bbox: dom_order_bbox(tbl.order),
                                });
                                *n += 1;
                            }
                        }
                        Ok(())
                    }),
                    // Capture landmark text (non-cell).
                    text!(LANDMARKS, move |t| {
                        if let Some(last) = p_tx.borrow_mut().last_mut() {
                            last.text.push_str(t.as_str());
                        }
                        Ok(())
                    }),
                    // Structured-data meta tags → document-level typed fields.
                    element!(META_FIELDS, move |el| {
                        // Key precedence: itemprop (microdata) > property
                        // (OpenGraph/RDFa) > name (classic/Dublin Core).
                        let key = el
                            .get_attribute("itemprop")
                            .or_else(|| el.get_attribute("property"))
                            .or_else(|| el.get_attribute("name"));
                        if let (Some(key), Some(value)) = (key, el.get_attribute("content")) {
                            let key = key.trim();
                            let value = value.trim();
                            if !key.is_empty() && !value.is_empty() {
                                let mut fs = f_meta.borrow_mut();
                                let idx = fs.len() as u16;
                                fs.push(TypedField {
                                    key: key.to_string(),
                                    value: value.to_string(),
                                    bbox: dom_order_bbox(idx),
                                    confidence: DOM_DECLARED_CONFIDENCE,
                                });
                            }
                        }
                        Ok(())
                    }),
                ],
                ..Settings::new()
            },
            |c: &[u8]| sink.extend_from_slice(c),
        );
        rw.write(content.as_bytes()).expect("lol_html rewrite");
        rw.end().expect("lol_html end");
    }

    let regions: Vec<Region> = pending
        .borrow()
        .iter()
        .map(|p| {
            let text = p.text.trim().to_string();
            Region {
                kind: p.kind,
                bbox: dom_order_bbox(p.order),
                reading_order: p.order,
                text: if text.is_empty() { None } else { Some(text) },
                cells: p.cells.clone(),
                children: vec![],
            }
        })
        .collect();

    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let content_sha256: [u8; 32] = hasher.finalize().into();

    let harvested_fields = fields.borrow().clone();

    DocIr {
        version: DOC_IR_VERSION.to_string(),
        source: Provenance::Dom,
        geometry: Geometry::DomOrder,
        content_sha256,
        mime: "text/html".to_string(),
        pages: vec![DocPage {
            number: 0,
            width: 0,
            height: 0,
            regions,
        }],
        fields: harvested_fields,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INVOICE: &str = r#"<!doctype html><html><body>
        <header>Acme GmbH</header>
        <nav>Home Kontakt</nav>
        <main>
          <p>Rechnung 2026-0042</p>
          <table>
            <tr><th>Pos</th><th>Betrag</th></tr>
            <tr><td>1</td><td>100,00</td></tr>
          </table>
        </main>
        <footer>Seite 1 von 1</footer>
    </body></html>"#;

    #[test]
    fn landmarks_become_regions_in_reading_order() {
        let ir = harvest(INVOICE);
        let regions = &ir.pages[0].regions;
        let kinds: Vec<RegionKind> = regions.iter().map(|r| r.kind).collect();
        assert_eq!(
            kinds,
            vec![
                RegionKind::Header,
                RegionKind::Nav,
                RegionKind::Main,
                RegionKind::Table,
                RegionKind::Footer,
            ],
            "HTML5 landmarks map to the closed vocab in document order"
        );
        for (i, r) in regions.iter().enumerate() {
            assert_eq!(
                r.reading_order as usize, i,
                "reading order is dense + ascending"
            );
        }
        assert_eq!(regions[0].text.as_deref(), Some("Acme GmbH"));
    }

    #[test]
    fn table_cells_are_harvested_onto_the_table_region() {
        let ir = harvest(INVOICE);
        let table = ir.pages[0]
            .regions
            .iter()
            .find(|r| r.kind == RegionKind::Table)
            .expect("a table region");
        let cell_text: Vec<&str> = table.cells.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(cell_text, vec!["Pos", "Betrag", "1", "100,00"]);
        // (row, col) address is dense in stream order (v1 heuristic).
        assert_eq!((table.cells[0].row, table.cells[0].col), (0, 0));
        assert_eq!((table.cells[1].row, table.cells[1].col), (0, 1));
    }

    #[test]
    fn meta_tags_become_typed_fields() {
        let html = r#"<!doctype html><html><head>
            <meta property="og:title" content="Rechnung 2026-0042">
            <meta name="author" content="Acme GmbH">
            <meta itemprop="iban" content="DE00 0000 0000 0000 0000 00">
            <meta name="empty" content="">
            <meta charset="utf-8">
          </head><body><main>x</main></body></html>"#;
        let ir = harvest(html);
        let got: Vec<(&str, &str)> = ir
            .fields
            .iter()
            .map(|f| (f.key.as_str(), f.value.as_str()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("og:title", "Rechnung 2026-0042"),
                ("author", "Acme GmbH"),
                ("iban", "DE00 0000 0000 0000 0000 00"),
            ],
            "meta name|property|itemprop with a non-empty content become typed fields; \
             empty-content and content-less meta are skipped"
        );
        assert!(
            ir.fields
                .iter()
                .all(|f| f.confidence == DOM_DECLARED_CONFIDENCE),
            "DOM-declared fields carry the source-scoped confidence"
        );
    }

    #[test]
    fn output_passes_the_source_agnostic_load_gate() {
        // The DOM producer's output must satisfy the SAME closed-vocab +
        // version gate an OCR producer's output does — the convergence proof.
        let ir = harvest(INVOICE);
        let json = ogar_doc_ir::to_json(&ir).expect("serialize");
        let back = ogar_doc_ir::from_json(&json).expect("the shared load gate accepts DOM output");
        assert_eq!(ir, back);
    }

    #[test]
    fn content_hash_is_the_convergence_key() {
        // Same bytes ⇒ same sha ⇒ one subtree (the P-XRETINA identity arm).
        let a = harvest(INVOICE);
        let b = harvest(INVOICE);
        assert_eq!(a.content_sha256, b.content_sha256);
        let c = harvest("<main>different</main>");
        assert_ne!(a.content_sha256, c.content_sha256);
    }
}
