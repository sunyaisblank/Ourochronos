import Std

/-!
An ideal, externally isolated ordinary-language core: 64-bit words, a stack,
three present-memory cells, two fixed vector handles, and unbounded vectors.
Heap lists are written top first (the reverse of the source vector order).
There is no ORACLE, epoch iteration, FFI, allocation bound, or instruction gas.

The source representation of counter n is [0, 1, ..., 1], with n one tokens.
Popping the zero sentinel distinguishes zero without converting an unbounded
length to a 64-bit word. The zero branch immediately restores that sentinel.

This proves translation of INC/DECJZ/HALT steps, not correctness of the Rust
compiler or VM. The imported two-counter universality result is a separate
mathematical premise; no theorem here asserts physical unbounded resources.
From formal/, check with the pinned toolchain: lean CounterMachine.lean.

Trusted base: Lean 4.29.1's kernel, its UInt64/list definitions, and the
standard logical principles reported by #print axioms (propext and Quot.sound).
There are no admitted proofs or user axioms. The imported universality premise
concerns a fixed finite universal INC/DECJZ program whose labels fit UInt64;
arbitrary input encodings reside in the unbounded natural-number counters.
Reference: Andrej Dudenhefner, Certified Decision Procedures for Two-Counter
Machines (FSCD 2022), §2/Theorem 6, doi:10.4230/LIPIcs.FSCD.2022.16.
Its CM2 increment at p embeds as inc c (p+1); its decrement at p targeting q
embeds as decjz c (p+1) q. This comparison assumes those finite labels fit
UInt64, and is a mathematical premise outside the proved translation.
The Rust conformance campaign is supporting finite evidence, not a proof of
compiler, dispatch, allocator, operating-system, or physical correctness.
-/

namespace Ourochronos

abbrev Word := UInt64

@[simp] theorem word_one_ne_zero : (1 : Word) ≠ 0 := by decide
@[simp] theorem word_two_ne_zero : (2 : Word) ≠ 0 := by decide
@[simp] theorem word_two_ne_one : (2 : Word) ≠ 1 := by decide

inductive Counter where
  | first | second
  deriving DecidableEq, Repr

inductive MachineInstruction where
  | inc (counter : Counter) (next : Word)
  | decjz (counter : Counter) (zero nonzero : Word)
  | halt
  deriving Repr

structure MachineState where
  pc : Word
  first : Nat
  second : Nat
  halted : Bool := false
  deriving Repr

def executeInstruction : MachineInstruction → MachineState → MachineState
  | .inc .first next, s => { s with first := s.first + 1, pc := next }
  | .inc .second next, s => { s with second := s.second + 1, pc := next }
  | .decjz .first zero nonzero, s =>
      match s.first with
      | 0 => { s with pc := zero }
      | n + 1 => { s with first := n, pc := nonzero }
  | .decjz .second zero nonzero, s =>
      match s.second with
      | 0 => { s with pc := zero }
      | n + 1 => { s with second := n, pc := nonzero }
  | .halt, s => { s with halted := true }

abbrev Program := List (Word × MachineInstruction)

def targets : MachineInstruction → List Word
  | .inc _ next => [next]
  | .decjz _ zero nonzero => [zero, nonzero]
  | .halt => []

/-- Declared finite counter programs have distinct labels and closed jumps.
    The simulation below also covers the explicit default-HALT convention. -/
def WellFormed (program : Program) : Prop :=
  (program.map Prod.fst).Nodup ∧
    ∀ entry ∈ program, ∀ target ∈ targets entry.2, target ∈ program.map Prod.fst

def lookup : Program → Word → MachineInstruction
  | [], _ => .halt
  | (label, instruction) :: rest, pc =>
      if pc = label then instruction else lookup rest pc

def machineStep (program : Program) (s : MachineState) : Option MachineState :=
  if s.halted then none else some (executeInstruction (lookup program s.pc) s)

structure Store where
  firstHandle : Word := 0
  secondHandle : Word := 1
  pc : Word
  deriving Repr

structure State where
  stack : List Word := []
  store : Store
  firstVector : List Word
  secondVector : List Word
  allocatedVectors : Nat := 2
  halted : Bool := false
  deriving Repr

inductive Primitive where
  | push (word : Word)
  | present | prophecy | vecNew | vecPush | vecPop | pop | eq | halt
  deriving Repr

inductive Command where
  | primitive (operation : Primitive)
  | branch (nonzero zero : List Command)
  /-- The ordinary structured `WHILE { 1 } { body }`, without gas accounting. -/
  | whileTrue (body : List Command)
  deriving Repr

structure Configuration where
  code : List Command
  state : State
  deriving Repr

def primitiveStep (operation : Primitive) (s : State) : Option State :=
  match operation, s.stack with
  | .push word, stack => some { s with stack := word :: stack }
  | .present, address :: stack =>
      if address = 0 then some { s with stack := s.store.firstHandle :: stack }
      else if address = 1 then some { s with stack := s.store.secondHandle :: stack }
      else if address = 2 then some { s with stack := s.store.pc :: stack }
      else none
  | .prophecy, address :: value :: stack =>
      if address = 0 then some { s with stack, store := { s.store with firstHandle := value } }
      else if address = 1 then
        some { s with stack, store := { s.store with secondHandle := value } }
      else if address = 2 then some { s with stack, store := { s.store with pc := value } }
      else none
  | .vecNew, stack =>
      match s.allocatedVectors with
      | 0 => some { s with stack := 0 :: stack, firstVector := [], allocatedVectors := 1 }
      | 1 => some { s with stack := 1 :: stack, secondVector := [], allocatedVectors := 2 }
      | _ => none
  | .vecPush, value :: handle :: stack =>
      if handle = 0 ∧ s.allocatedVectors > 0 then
        some { s with stack := handle :: stack, firstVector := value :: s.firstVector }
      else if handle = 1 ∧ s.allocatedVectors > 1 then
        some { s with stack := handle :: stack, secondVector := value :: s.secondVector }
      else none
  | .vecPop, handle :: stack =>
      if handle = 0 ∧ s.allocatedVectors > 0 then
        match s.firstVector with
        | [] => none
        | value :: vector => some { s with stack := value :: handle :: stack, firstVector := vector }
      else if handle = 1 ∧ s.allocatedVectors > 1 then
        match s.secondVector with
        | [] => none
        | value :: vector => some { s with stack := value :: handle :: stack, secondVector := vector }
      else none
  | .pop, _ :: stack => some { s with stack }
  | .eq, a :: b :: stack => some { s with stack := (if a = b then 1 else 0) :: stack }
  | .halt, _ => some { s with halted := true }
  | _, _ => none

def step (configuration : Configuration) : Option Configuration :=
  if configuration.state.halted then none else
  match configuration.code with
  | [] => none
  | .primitive operation :: rest =>
      (primitiveStep operation configuration.state).map fun s => ⟨rest, s⟩
  | .branch nonzero zero :: rest =>
      match configuration.state.stack with
      | [] => none
      | condition :: stack => some ⟨(if condition = 0 then zero else nonzero) ++ rest,
                                     { configuration.state with stack }⟩
  | .whileTrue body :: rest =>
      some ⟨body ++ [.whileTrue body] ++ rest, configuration.state⟩

inductive Steps : Configuration → Configuration → Prop where
  | refl (s) : Steps s s
  | cons {s t u} : step s = some t → Steps t u → Steps s u

def run : Nat → Configuration → Option Configuration
  | 0, c => if c.code.isEmpty then some c else none
  | n + 1, c => if c.code.isEmpty then some c else (step c).bind (run n)

theorem run_sound {fuel : Nat} {start finish : Configuration}
    (h : run fuel start = some finish) : Steps start finish := by
  induction fuel generalizing start with
  | zero =>
      simp only [run] at h
      split at h
      · cases h
        exact .refl _
      · contradiction
  | succ fuel ih =>
      simp only [run] at h
      split at h
      · cases h
        exact .refl _
      · cases hs : step start with
        | none => simp [hs] at h
        | some next =>
            simp [hs] at h
            exact .cons hs (ih h)

def advance : Nat → Configuration → Option Configuration
  | 0, c => some c
  | n + 1, c => (step c).bind (advance n)

theorem advance_sound {fuel : Nat} {start finish : Configuration}
    (h : advance fuel start = some finish) : Steps start finish := by
  induction fuel generalizing start with
  | zero =>
      cases h
      exact .refl _
  | succ fuel ih =>
      cases hs : step start with
      | none => simp [advance, hs] at h
      | some next =>
          simp [advance, hs] at h
          exact .cons hs (ih h)

theorem Steps.trans {a b c : Configuration} (first : Steps a b) (second : Steps b c) :
    Steps a c := by
  induction first with
  | refl => exact second
  | cons h _ ih => exact .cons h (ih second)

def tokens : Nat → List Word
  | 0 => [0]
  | n + 1 => 1 :: tokens n

theorem tokens_length (n : Nat) : (tokens n).length = n + 1 := by
  induction n with
  | zero => rfl
  | succ n ih => simp [tokens, ih, Nat.add_assoc]

theorem tokens_injective {a b : Nat} (h : tokens a = tokens b) : a = b := by
  have lengths := congrArg List.length h
  simpa only [tokens_length, Nat.add_right_cancel_iff] using lengths

def encode (s : MachineState) : State :=
  { store := ⟨0, 1, s.pc⟩, firstVector := tokens s.first,
    secondVector := tokens s.second, halted := s.halted }

/-- Counter readout consists of heap token sequences, never a length cast to
    a machine word. Rust conformance inspects these same exported vectors. -/
def observedCounters (s : State) : List Word × List Word :=
  (s.firstVector.reverse, s.secondVector.reverse)

theorem encoded_counter_observation (s : MachineState) :
    observedCounters (encode s) = ((tokens s.first).reverse, (tokens s.second).reverse) := by
  rfl

theorem encoded_readout_injective {s t : MachineState}
    (equal : observedCounters (encode s) = observedCounters (encode t)) :
    s.first = t.first ∧ s.second = t.second := by
  constructor
  · apply tokens_injective
    have first := congrArg (fun pair => pair.1.reverse) equal
    simpa [observedCounters, encode] using first
  · apply tokens_injective
    have second := congrArg (fun pair => pair.2.reverse) equal
    simpa [observedCounters, encode] using second

def handle : Counter → Word
  | .first => 0
  | .second => 1

def op := Command.primitive

def pushToken (counter : Counter) : List Command :=
  [op (.push (handle counter)), op .present, op (.push 1), op .vecPush, op .pop]

def setPC (next : Word) : List Command :=
  [op (.push next), op (.push 2), op .prophecy]

/-- Exact source header: VEC_NEW 0 VEC_PUSH 0 PROPHECY;
    VEC_NEW 0 VEC_PUSH 1 PROPHECY; pc 2 PROPHECY. -/
def zeroCounterHeader (pc : Word) : List Command :=
  [op .vecNew, op (.push 0), op .vecPush, op (.push 0), op .prophecy,
   op .vecNew, op (.push 0), op .vecPush, op (.push 1), op .prophecy] ++ setPC pc

def emptyState : State :=
  { store := ⟨0, 0, 0⟩, firstVector := [], secondVector := [], allocatedVectors := 0 }

theorem zero_counter_header_correct (pc : Word) :
    run 20 ⟨zeroCounterHeader pc, emptyState⟩ =
      some ⟨[], encode ⟨pc, 0, 0, false⟩⟩ := by
  rfl

def emit : MachineInstruction → List Command
  | .inc counter next =>
      pushToken counter ++ setPC next
  | .decjz counter zero nonzero =>
      [op (.push (handle counter)), op .present, op .vecPop,
       .branch ([op .pop] ++ setPC nonzero)
               ([op (.push 0), op .vecPush, op .pop] ++ setPC zero)]
  | .halt => [op .halt]

theorem emitted_instruction_correct (instruction : MachineInstruction) (s : MachineState)
    (running : s.halted = false) :
    run 12 ⟨emit instruction, encode s⟩ =
      some ⟨[], encode (executeInstruction instruction s)⟩ := by
  cases s with
  | mk pc first second halted =>
      cases halted
      · cases instruction with
        | inc counter next => cases counter <;> rfl
        | decjz counter zero nonzero =>
            cases counter
            · cases first <;> rfl
            · cases second <;> rfl
        | halt => rfl
      · simp at running

theorem instruction_simulation (instruction : MachineInstruction) (s : MachineState)
    (running : s.halted = false) :
    Steps ⟨emit instruction, encode s⟩ ⟨[], encode (executeInstruction instruction s)⟩ :=
  run_sound (emitted_instruction_correct instruction s running)

def selection (label : Word) (yes no : List Command) : List Command :=
  [op (.push 2), op .present, op (.push label), op .eq, .branch yes no]

def emitDispatch : Program → List Command
  | [] => emit .halt
  | (label, instruction) :: rest => selection label (emit instruction) (emitDispatch rest)

theorem selection_correct (label : Word) (yes no : List Command) (s : MachineState)
    (running : s.halted = false) :
    advance 5 ⟨selection label yes no, encode s⟩ =
      some ⟨if s.pc = label then yes else no, encode s⟩ := by
  cases s with
  | mk pc first second halted =>
      cases halted
      · by_cases h : pc = label
        · subst label
          simp [advance, selection, op, step, primitiveStep, encode]
        · have reverse : label ≠ pc := Ne.symm h
          simp [advance, selection, op, step, primitiveStep, encode, h, reverse]
      · simp at running

theorem dispatch_simulation (program : Program) (s : MachineState)
    (running : s.halted = false) :
    Steps ⟨emitDispatch program, encode s⟩
      ⟨[], encode (executeInstruction (lookup program s.pc) s)⟩ := by
  induction program with
  | nil => exact instruction_simulation .halt s running
  | cons entry rest ih =>
      rcases entry with ⟨label, instruction⟩
      have selected := advance_sound (selection_correct label (emit instruction)
        (emitDispatch rest) s running)
      by_cases h : s.pc = label
      · simp only [if_pos h] at selected
        simpa only [emitDispatch, lookup, if_pos h] using
          selected.trans (instruction_simulation instruction s running)
      · simp only [if_neg h] at selected
        simpa only [emitDispatch, lookup, if_neg h] using selected.trans ih

def withContinuation (c : Configuration) (tail : List Command) : Configuration :=
  ⟨c.code ++ tail, c.state⟩

theorem step_with_continuation {a b : Configuration} (h : step a = some b)
    (tail : List Command) :
    step (withContinuation a tail) = some (withContinuation b tail) := by
  rcases a with ⟨code, state⟩
  cases halted : state.halted
  · cases code with
    | nil => simp [step, halted] at h
    | cons command rest =>
        cases command with
        | primitive operation =>
            cases primitive : primitiveStep operation state with
            | none => simp [step, halted, primitive] at h
            | some next =>
                simp [step, halted, primitive] at h
                cases h
                simp [step, withContinuation, halted, primitive]
        | branch yes no =>
            cases stack : state.stack with
            | nil => simp [step, halted, stack] at h
            | cons condition stackRest =>
                simp [step, halted, stack] at h
                cases h
                simp [step, withContinuation, halted, stack, List.append_assoc]
        | whileTrue body =>
            simp [step, halted] at h
            cases h
            simp [step, withContinuation, halted, List.append_assoc]
  · simp [step, halted] at h

theorem Steps.with_continuation {a b : Configuration} (h : Steps a b)
    (tail : List Command) : Steps (withContinuation a tail) (withContinuation b tail) := by
  induction h with
  | refl => exact .refl _
  | cons next _ ih => exact .cons (step_with_continuation next tail) ih

def inputTokens (counter : Counter) : Nat → List Command
  | 0 => []
  | n + 1 => pushToken counter ++ inputTokens counter n

def addInput (counter : Counter) (count : Nat) (s : MachineState) : MachineState :=
  match counter with
  | .first => { s with first := s.first + count }
  | .second => { s with second := s.second + count }

theorem push_token_correct (counter : Counter) (s : MachineState)
    (running : s.halted = false) :
    run 5 ⟨pushToken counter, encode s⟩ =
      some ⟨[], encode (executeInstruction (.inc counter s.pc) s)⟩ := by
  cases s with
  | mk pc first second halted =>
      cases halted
      · cases counter <;> rfl
      · simp at running

theorem input_tokens_simulation (counter : Counter) (count : Nat) (s : MachineState)
    (running : s.halted = false) :
    Steps ⟨inputTokens counter count, encode s⟩ ⟨[], encode (addInput counter count s)⟩ := by
  induction count generalizing s with
  | zero =>
      cases counter <;> simpa [inputTokens, addInput] using
        (Steps.refl (⟨[], encode s⟩ : Configuration))
  | succ n ih =>
      have first := (run_sound (push_token_correct counter s running)).with_continuation
        (inputTokens counter n)
      have stillRunning : (executeInstruction (.inc counter s.pc) s).halted = false := by
        cases counter <;> simpa [executeInstruction] using running
      have rest := ih (executeInstruction (.inc counter s.pc) s) stillRunning
      have completed := first.trans rest
      cases counter <;>
        simpa [inputTokens, withContinuation, addInput, executeInstruction,
          Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using completed

def inputHeader (pc : Word) (first second : Nat) : List Command :=
  zeroCounterHeader pc ++ inputTokens .first first ++ inputTokens .second second

theorem initialization_correct (pc : Word) (first second : Nat) :
    Steps ⟨inputHeader pc first second, emptyState⟩ ⟨[], encode ⟨pc, first, second, false⟩⟩ := by
  have header := (run_sound (zero_counter_header_correct pc)).with_continuation
    (inputTokens .first first ++ inputTokens .second second)
  have firstInput := (input_tokens_simulation .first first ⟨pc, 0, 0, false⟩ rfl).with_continuation
    (inputTokens .second second)
  have secondInput := input_tokens_simulation .second second ⟨pc, first, 0, false⟩ rfl
  simp only [withContinuation, addInput, List.nil_append, Nat.zero_add] at header firstInput secondInput
  simpa [inputHeader, withContinuation, addInput, List.append_assoc] using
    header.trans (firstInput.trans secondInput)

def translated (program : Program) (s : MachineState) : Configuration :=
  ⟨[.whileTrue (emitDispatch program)], encode s⟩

theorem machine_step_simulation (program : Program) (s t : MachineState)
    (machine : machineStep program s = some t) :
    Steps (translated program s) (translated program t) := by
  have running : s.halted = false := by
    cases halted : s.halted
    · rfl
    · simp [machineStep, halted] at machine
  have result : executeInstruction (lookup program s.pc) s = t := by
    simpa [machineStep, running] using machine
  have body := (dispatch_simulation program s running).with_continuation
    [.whileTrue (emitDispatch program)]
  apply Steps.cons (t := withContinuation ⟨emitDispatch program, encode s⟩
      [.whileTrue (emitDispatch program)])
  · simp [step, translated, withContinuation, encode, running]
  · simpa [translated, withContinuation, result] using body

inductive MachineSteps (program : Program) : MachineState → MachineState → Prop where
  | refl (s) : MachineSteps program s s
  | cons {s t u} : machineStep program s = some t →
      MachineSteps program t u → MachineSteps program s u

theorem finite_trace_simulation {program : Program} {s t : MachineState}
    (trace : MachineSteps program s t) : Steps (translated program s) (translated program t) := by
  induction trace with
  | refl => exact .refl _
  | cons next _ ih => exact (machine_step_simulation program _ _ next).trans ih

theorem initialized_trace_simulation {program : Program} {pc : Word} {first second : Nat}
    {finish : MachineState} (trace : MachineSteps program ⟨pc, first, second, false⟩ finish) :
    Steps ⟨inputHeader pc first second ++ [.whileTrue (emitDispatch program)], emptyState⟩
      (translated program finish) := by
  have initialized := (initialization_correct pc first second).with_continuation
    [.whileTrue (emitDispatch program)]
  exact initialized.trans (finite_trace_simulation trace)

theorem halted_translation_is_terminal (program : Program) (s : MachineState)
    (halted : s.halted = true) : step (translated program s) = none := by
  simp [step, translated, encode, halted]

#print axioms instruction_simulation
#print axioms finite_trace_simulation
#print axioms tokens_injective
#print axioms zero_counter_header_correct
#print axioms initialized_trace_simulation
#print axioms encoded_readout_injective

end Ourochronos
