//! Deterministic fuzz-style acceptance checks for untrusted compiler inputs.

use std::panic::{catch_unwind, AssertUnwindSafe};

fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

#[test]
fn bounded_artifact_decoders_never_panic_on_adversarial_bytes() {
    let mut seed = 0x4f55_524f_4348_524fu64;
    for case in 0..4_096usize {
        let length = (next(&mut seed) as usize) % 2_048;
        let mut bytes = vec![0u8; length];
        for byte in &mut bytes {
            *byte = next(&mut seed) as u8;
        }
        // Plant real magics frequently so fuzzing reaches bounded length and
        // table decoders rather than stopping at the first byte.
        if bytes.len() >= 8 {
            let magic = match case % 3 {
                0 => b"OUROBC\0\0".as_slice(),
                1 => b"OUROOBJ\0".as_slice(),
                _ => b"OUROPK\0\0".as_slice(),
            };
            bytes[..8].copy_from_slice(magic);
        }

        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _ = ourochronos::BytecodeProgram::from_bytes(&bytes);
                let _ = ourochronos::ObjectModule::from_bytes(&bytes);
                let _ = ourochronos::PortablePackage::from_bytes(&bytes);
                let _ = ourochronos::embedded_package(&bytes);
            }))
            .is_ok(),
            "decoder panic for deterministic corpus case {case}"
        );
    }
}

#[test]
fn located_lexer_parser_never_panics_on_generated_utf8() {
    let atoms = [
        "λ",
        "🜂",
        "{",
        "}",
        "[",
        "]",
        "\"",
        "#",
        "0",
        "18446744073709551616",
        "TEMPORAL",
        "PROCEDURE",
        "IMPORT",
        "BOGUS",
        "\n",
        " ",
        "\\",
    ];
    let mut seed = 0x5350_414e_4655_5a5au64;
    for case in 0..2_048usize {
        let count = (next(&mut seed) as usize % 48) + 1;
        let mut source = String::new();
        for _ in 0..count {
            source.push_str(atoms[next(&mut seed) as usize % atoms.len()]);
        }
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                if let Ok(located_tokens) =
                    ourochronos::lex(ourochronos::SourceId::new(case), &source)
                {
                    let tokens = located_tokens
                        .iter()
                        .map(|token| token.token.clone())
                        .collect::<Vec<_>>();
                    let spans = located_tokens
                        .iter()
                        .map(|token| token.span)
                        .collect::<Vec<_>>();
                    if let Ok(mut parser) =
                        ourochronos::Parser::new_with_source_spans(&tokens, &spans)
                    {
                        let _ = parser.parse_located_program();
                    }
                }
            }))
            .is_ok(),
            "frontend panic for deterministic UTF-8 corpus case {case}: {source:?}"
        );
    }
}
