//! Explicit resource and complexity contracts for Ourochronos programs.
//!
//! A concrete run profile is measured independently from an asymptotic
//! program-family claim. The latter records the assumptions of the
//! Aaronson--Watrous model; it does not pretend those semantic obligations
//! have been mechanically proved.

use std::fmt;

use crate::ast::Program;

/// Exact finite resource bounds configured for one executable instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcreteResourceProfile {
    pub temporal_cells: usize,
    pub cell_bits: usize,
    pub temporal_state_bits: u128,
    pub address_bits: u32,
    pub max_instructions_per_epoch: u64,
    pub max_epochs: usize,
    pub temporal_operations_in_source: usize,
    pub semantics: &'static str,
}

impl ConcreteResourceProfile {
    pub fn for_program(
        program: &Program,
        temporal_cells: usize,
        max_instructions_per_epoch: u64,
        max_epochs: usize,
        semantics: &'static str,
    ) -> Self {
        let address_bits = if temporal_cells <= 1 {
            0
        } else {
            usize::BITS - (temporal_cells - 1).leading_zeros()
        };
        Self {
            temporal_cells,
            cell_bits: 64,
            temporal_state_bits: temporal_cells as u128 * 64,
            address_bits,
            max_instructions_per_epoch,
            max_epochs,
            temporal_operations_in_source: program.temporal_op_count(),
            semantics,
        }
    }
}

impl fmt::Display for ConcreteResourceProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Concrete Resource Profile ===")?;
        writeln!(
            f,
            "Temporal memory: {} cells x {} bits = {} state bits",
            self.temporal_cells, self.cell_bits, self.temporal_state_bits
        )?;
        writeln!(f, "Address bits required: {}", self.address_bits)?;
        writeln!(
            f,
            "Temporal opcodes in source: {}",
            self.temporal_operations_in_source
        )?;
        writeln!(
            f,
            "Epoch gas bound: {} instructions",
            self.max_instructions_per_epoch
        )?;
        writeln!(f, "Orbit-search bound: {} epochs", self.max_epochs)?;
        writeln!(f, "Consistency semantics: {}", self.semantics)?;
        writeln!(
            f,
            "Asymptotic status: one finite instance; not by itself a PSPACE-family proof"
        )
    }
}

/// A polynomial upper bound `coefficient * n^degree + additive`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolynomialBound {
    pub coefficient: u64,
    pub degree: u32,
    pub additive: u64,
}

impl PolynomialBound {
    pub fn evaluate(&self, input_bits: u64) -> Option<u128> {
        let power = (input_bits as u128).checked_pow(self.degree)?;
        (self.coefficient as u128)
            .checked_mul(power)?
            .checked_add(self.additive as u128)
    }
}

/// Declared obligations for a polynomial-size Deutschian CTC program family.
///
/// `declared_eligible` means the author has supplied every assumption required
/// by this contract. It is intentionally not called a proof: uniformity,
/// totality, and readout invariance require separate verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PspaceFamilyContract {
    pub name: String,
    pub ctc_cells: PolynomialBound,
    pub chronology_respecting_bits: PolynomialBound,
    pub transition_steps: PolynomialBound,
    pub polynomial_time_uniform: bool,
    pub total_transition: bool,
    pub all_fixed_points_agree: bool,
    pub ideal_deutsch_selector: bool,
    pub effects_frozen_or_modeled: bool,
}

impl PspaceFamilyContract {
    /// Missing Aaronson--Watrous assumptions, in stable explanatory order.
    pub fn missing_obligations(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if !self.polynomial_time_uniform {
            missing.push("polynomial-time uniform circuit generation");
        }
        if !self.total_transition {
            missing.push("a total deterministic/stochastic transition");
        }
        if !self.all_fixed_points_agree {
            missing.push("the same decision readout on every fixed point/recurrent class");
        }
        if !self.ideal_deutsch_selector {
            missing.push("an ideal Deutsch stationary-state selector");
        }
        if !self.effects_frozen_or_modeled {
            missing.push("all external inputs/effects frozen or included in the transition");
        }
        missing
    }

    pub fn declared_eligible(&self) -> bool {
        self.missing_obligations().is_empty()
    }

    /// Check that this concrete instance fits its declared polynomial bounds.
    pub fn instance_fits(
        &self,
        input_bits: u64,
        ctc_cells: usize,
        chronology_respecting_bits: usize,
        transition_steps: u64,
    ) -> bool {
        let Some(cell_bound) = self.ctc_cells.evaluate(input_bits) else {
            return false;
        };
        let Some(cr_bound) = self.chronology_respecting_bits.evaluate(input_bits) else {
            return false;
        };
        let Some(step_bound) = self.transition_steps.evaluate(input_bits) else {
            return false;
        };
        ctc_cells as u128 <= cell_bound
            && chronology_respecting_bits as u128 <= cr_bound
            && transition_steps as u128 <= step_bound
    }
}

impl From<&crate::ast::PspaceFamilyDeclaration> for PspaceFamilyContract {
    fn from(declaration: &crate::ast::PspaceFamilyDeclaration) -> Self {
        let polynomial = |bound: crate::ast::PolynomialDeclaration| PolynomialBound {
            coefficient: bound.coefficient,
            degree: bound.degree,
            additive: bound.additive,
        };
        Self {
            name: declaration.name.clone(),
            ctc_cells: polynomial(declaration.ctc_cells),
            chronology_respecting_bits: polynomial(declaration.chronology_respecting_bits),
            transition_steps: polynomial(declaration.transition_steps),
            polynomial_time_uniform: declaration.polynomial_time_uniform,
            total_transition: declaration.total_transition,
            all_fixed_points_agree: declaration.all_fixed_points_agree,
            ideal_deutsch_selector: declaration.ideal_deutsch_selector,
            effects_frozen_or_modeled: declaration.effects_frozen_or_modeled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn concrete_profile_reports_configured_width() {
        let program = parse("0 ORACLE 0 PROPHECY").unwrap();
        let profile = ConcreteResourceProfile::for_program(&program, 70_000, 1_000, 20, "Deutsch");
        assert_eq!(profile.temporal_state_bits, 4_480_000);
        assert_eq!(profile.address_bits, 17);
        assert_eq!(profile.temporal_operations_in_source, 2);
    }

    #[test]
    fn family_contract_exposes_unproved_obligations() {
        let contract = PspaceFamilyContract {
            name: "test".into(),
            ctc_cells: PolynomialBound {
                coefficient: 2,
                degree: 2,
                additive: 1,
            },
            chronology_respecting_bits: PolynomialBound {
                coefficient: 1,
                degree: 1,
                additive: 0,
            },
            transition_steps: PolynomialBound {
                coefficient: 4,
                degree: 3,
                additive: 0,
            },
            polynomial_time_uniform: true,
            total_transition: true,
            all_fixed_points_agree: false,
            ideal_deutsch_selector: true,
            effects_frozen_or_modeled: true,
        };
        assert!(!contract.declared_eligible());
        assert_eq!(contract.missing_obligations().len(), 1);
        assert!(contract.instance_fits(10, 200, 10, 4_000));
    }

    #[test]
    fn source_family_declaration_round_trips_into_contract() {
        let program = parse(
            "FAMILY demo {\n\
             CTC_CELLS POLY 2 2 1;\n\
             CHRONOLOGY_BITS POLY 1 1 0;\n\
             TRANSITION_STEPS POLY 4 3 0;\n\
             UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN;\n\
             }\n0 ORACLE 0 PROPHECY",
        )
        .unwrap();
        let declaration = program.family_declaration.as_ref().unwrap();
        let contract = PspaceFamilyContract::from(declaration);
        assert_eq!(contract.name, "demo");
        assert!(contract.declared_eligible());
        assert_eq!(contract.ctc_cells.evaluate(10), Some(201));
    }

    #[test]
    fn incomplete_family_declaration_is_rejected() {
        let error = parse("FAMILY bad { CTC_CELLS POLY 1 1 0; }").unwrap_err();
        assert!(error.contains("CHRONOLOGY_BITS"));
    }
}
