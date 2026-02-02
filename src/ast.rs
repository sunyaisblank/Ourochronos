//! Abstract Syntax Tree for OUROCHRONOS.
//!
//! OUROCHRONOS uses a stack-based execution model with structured control flow.
//! Programs consist of statements that manipulate a value stack, present memory,
//! and can read from anamnesis (the temporal oracle).

use crate::core::Value;

/// All operations available in OUROCHRONOS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCode {
    // ═══════════════════════════════════════════════════════════════════
    // Stack Manipulation
    // ═══════════════════════════════════════════════════════════════════
    
    /// No operation.
    Nop,
    
    /// Halt execution of the current epoch.
    /// Stack: ( -- )
    Halt,
    
    /// Pop and discard the top of stack.
    /// Stack: ( a -- )
    Pop,
    
    /// Duplicate the top of stack.
    /// Stack: ( a -- a a )
    Dup,
    
    /// Swap the top two elements.
    /// Stack: ( a b -- b a )
    Swap,
    
    /// Copy the second element to the top.
    /// Stack: ( a b -- a b a )
    Over,
    
    /// Rotate the top three elements.
    /// Stack: ( a b c -- b c a )
    Rot,
    
    /// Push the current stack depth.
    /// Stack: ( -- n )
    Depth,

    /// Pick the nth element from deep in the stack and copy it to top.
    /// 0 PICK is equivalent to DUP. 1 PICK is equivalent to OVER.
    /// Stack: ( n -- v )
    Pick,

    /// Roll the nth element to the top, shifting others down.
    /// 0 ROLL is Nop. 1 ROLL is SWAP.
    /// Stack: ( n -- )
    Roll,

    /// Reverse the top n elements.
    /// Stack: ( n -- )
    Reverse,

    /// Execute the quotation (code block) at the given index.
    /// Stack: ( quote_id -- )
    Exec,

    /// Execute a quotation while hiding the top element of the stack.
    /// Stack: ( x quote_id -- x )
    Dip,

    /// Keep x, execute quote, restore x (below result).
    /// Stack: ( x quote -- ... x )
    Keep,
    
    /// Bi-call: Apply p to x, then q to x.
    /// Stack: ( x p q -- ... ... )
    Bi,
    
    /// Recursive combinator: Execute quote with quote on stack.
    /// Stack: ( quote -- ... )
    Rec,
    
    // ════ String Operations ════
    
    // ════ String Operations ════
    
    // ════ String Operations ════
    
    /// Reverse a string (len-suffixed sequence).
    /// Stack: ( chars... len -- reversed_chars... len )
    StrRev,
    
    /// Concatenate two strings.
    /// Stack: ( c1.. len1 c2.. len2 -- c1..c2.. (len1+len2) )
    StrCat,
    
    /// Split string by delimiter char.
    /// Stack: ( chars... len delim_char -- s1 s2 .. sn count )
    StrSplit,

    // ═══════════════════════════════════════════════════════════════════
    // Arithmetic (modular, wrapping at 2^64)
    // ═══════════════════════════════════════════════════════════════════
    
    /// Addition.
    /// Stack: ( a b -- a+b )
    Add,
    
    /// Subtraction.
    /// Stack: ( a b -- a-b )
    Sub,
    
    /// Multiplication.
    /// Stack: ( a b -- a*b )
    Mul,
    
    /// Division (returns 0 if divisor is 0).
    /// Stack: ( a b -- a/b )
    Div,
    
    /// Modulo (returns 0 if divisor is 0).
    /// Stack: ( a b -- a%b )
    Mod,
    
    /// Negation (two's complement).
    /// Stack: ( a -- -a )
    Neg,

    /// Absolute value (signed interpretation).
    /// Stack: ( a -- |a| )
    Abs,

    /// Minimum of two values.
    /// Stack: ( a b -- min(a,b) )
    Min,

    /// Maximum of two values.
    /// Stack: ( a b -- max(a,b) )
    Max,

    /// Sign of value (signed interpretation): -1, 0, or 1.
    /// Stack: ( a -- sign(a) )
    Sign,

    /// Assertion.
    /// Pops length, chars, then condition. panics if condition is 0.
    /// Stack: ( cond chars... len -- )
    Assert,
    
    // ═══════════════════════════════════════════════════════════════════
    // Bitwise Logic
    // ═══════════════════════════════════════════════════════════════════
    
    /// Bitwise NOT.
    /// Stack: ( a -- ~a )
    Not,
    
    /// Bitwise AND.
    /// Stack: ( a b -- a&b )
    And,
    
    /// Bitwise OR.
    /// Stack: ( a b -- a|b )
    Or,
    
    /// Bitwise XOR.
    /// Stack: ( a b -- a^b )
    Xor,
    
    /// Left shift.
    /// Stack: ( a n -- a<<n )
    Shl,
    
    /// Right shift (logical).
    /// Stack: ( a n -- a>>n )
    Shr,
    
    // ═══════════════════════════════════════════════════════════════════
    // Comparison (result: 1 if true, 0 if false)
    // ═══════════════════════════════════════════════════════════════════
    
    /// Equal.
    /// Stack: ( a b -- a==b )
    Eq,
    
    /// Not equal.
    /// Stack: ( a b -- a!=b )
    Neq,
    
    /// Less than.
    /// Stack: ( a b -- a<b )
    Lt,
    
    /// Greater than.
    /// Stack: ( a b -- a>b )
    Gt,
    
    /// Less than or equal.
    /// Stack: ( a b -- a<=b )
    Lte,
    
    /// Greater than or equal.
    /// Stack: ( a b -- a>=b )
    Gte,

    // ═══════════════════════════════════════════════════════════════════
    // Signed Comparison (for two's complement interpretation)
    // ═══════════════════════════════════════════════════════════════════

    /// Signed less than.
    /// Stack: ( a b -- a<b ) where a,b are treated as signed
    Slt,

    /// Signed greater than.
    /// Stack: ( a b -- a>b ) where a,b are treated as signed
    Sgt,

    /// Signed less than or equal.
    /// Stack: ( a b -- a<=b ) where a,b are treated as signed
    Slte,

    /// Signed greater than or equal.
    /// Stack: ( a b -- a>=b ) where a,b are treated as signed
    Sgte,

    // ═══════════════════════════════════════════════════════════════════
    // Temporal Operations (the core of OUROCHRONOS)
    // ═══════════════════════════════════════════════════════════════════
    
    /// ORACLE: Read from anamnesis (the future).
    /// Pops an address, pushes the value from Anamnesis[address].
    /// This is the mechanism for receiving information from the future.
    /// Stack: ( addr -- A[addr] )
    Oracle,
    
    /// PROPHECY: Write to present (fulfilling the future).
    /// Pops value and address, writes value to Present[address].
    /// This is the mechanism for sending information to the past.
    /// Stack: ( value addr -- )
    Prophecy,
    
    /// PRESENT_READ: Read from present memory (current epoch).
    /// Stack: ( addr -- P[addr] )
    PresentRead,
    
    /// PARADOX: Signal explicit inconsistency.
    /// Terminates the current epoch as paradoxical.
    /// In fixed-point search, this path is rejected.
    /// Stack: ( -- )
    Paradox,
    
    // ═══════════════════════════════════════════════════════════════════
    // Array/Memory Operations
    // ═══════════════════════════════════════════════════════════════════
    
    /// Pack n values into contiguous memory starting at base address.
    /// Stack: ( v1 v2 ... vn base n -- )
    Pack,
    
    /// Unpack n values from contiguous memory at base address.
    /// Stack: ( base n -- v1 v2 ... vn )
    Unpack,
    
    /// Read from memory at computed index: base + index.
    /// Stack: ( base index -- P[base+index] )
    Index,
    
    /// Store to memory at computed index: base + index.
    /// Stack: ( value base index -- )
    Store,
    
    // ═══════════════════════════════════════════════════════════════════
    // I/O Operations
    // ═══════════════════════════════════════════════════════════════════

    /// Read a value from input.
    /// Stack: ( -- value )
    Input,

    /// Output a value.
    /// Stack: ( value -- )
    Output,

    /// Output a value as a character.
    /// Stack: ( char -- )
    Emit,

    // ═══════════════════════════════════════════════════════════════════
    // Vector Operations (Dynamic Arrays)
    // ═══════════════════════════════════════════════════════════════════

    /// Create a new empty vector.
    /// Returns a handle (index into vector storage).
    /// Stack: ( -- vec_handle )
    VecNew,

    /// Push a value onto the end of a vector.
    /// Stack: ( vec value -- vec )
    VecPush,

    /// Pop a value from the end of a vector.
    /// Stack: ( vec -- vec value )
    VecPop,

    /// Get a value at index from a vector.
    /// Stack: ( vec index -- vec value )
    VecGet,

    /// Set a value at index in a vector.
    /// Stack: ( vec value index -- vec )
    VecSet,

    /// Get the length of a vector.
    /// Stack: ( vec -- vec length )
    VecLen,

    // ═══════════════════════════════════════════════════════════════════
    // Hash Table Operations
    // ═══════════════════════════════════════════════════════════════════

    /// Create a new empty hash table.
    /// Returns a handle (index into hash storage).
    /// Stack: ( -- hash_handle )
    HashNew,

    /// Insert or update a key-value pair in a hash table.
    /// Stack: ( hash key value -- hash )
    HashPut,

    /// Get a value by key from a hash table.
    /// Returns the value and a found flag (1 if found, 0 if not).
    /// Stack: ( hash key -- hash value found )
    HashGet,

    /// Delete a key from a hash table.
    /// Stack: ( hash key -- hash )
    HashDel,

    /// Check if a key exists in a hash table.
    /// Stack: ( hash key -- hash found )
    HashHas,

    /// Get the number of entries in a hash table.
    /// Stack: ( hash -- hash count )
    HashLen,

    // ═══════════════════════════════════════════════════════════════════
    // Set Operations
    // ═══════════════════════════════════════════════════════════════════

    /// Create a new empty set.
    /// Returns a handle (index into set storage).
    /// Stack: ( -- set_handle )
    SetNew,

    /// Add a value to a set.
    /// Stack: ( set value -- set )
    SetAdd,

    /// Check if a value is in a set.
    /// Stack: ( set value -- set found )
    SetHas,

    /// Remove a value from a set.
    /// Stack: ( set value -- set )
    SetDel,

    /// Get the number of elements in a set.
    /// Stack: ( set -- set count )
    SetLen,

    // ═══════════════════════════════════════════════════════════════════
    // FFI Operations (Foreign Function Interface)
    // ═══════════════════════════════════════════════════════════════════

    /// Call an FFI function by ID.
    /// Stack effect depends on the function signature.
    /// Stack: ( args... ffi_id -- results... )
    FFICall,

    /// Call an FFI function by name (resolved at runtime).
    /// Stack: ( args... name_handle -- results... )
    FFICallNamed,

    // ═══════════════════════════════════════════════════════════════════
    // File I/O Operations
    // ═══════════════════════════════════════════════════════════════════

    /// Open a file.
    /// Stack: ( path_handle mode -- file_handle )
    /// Mode: 1=read, 2=write, 4=append, 8=create, 16=truncate
    FileOpen,

    /// Read from a file into a buffer.
    /// Stack: ( file_handle max_bytes -- file_handle buffer_handle bytes_read )
    FileRead,

    /// Write buffer contents to a file.
    /// Stack: ( file_handle buffer_handle -- file_handle bytes_written )
    FileWrite,

    /// Seek to a position in a file.
    /// Stack: ( file_handle offset origin -- file_handle new_position )
    /// Origin: 0=start, 1=current, 2=end
    FileSeek,

    /// Flush file buffers to disk.
    /// Stack: ( file_handle -- file_handle )
    FileFlush,

    /// Close a file.
    /// Stack: ( file_handle -- )
    FileClose,

    /// Check if a file exists.
    /// Stack: ( path_handle -- exists )
    FileExists,

    /// Get file size.
    /// Stack: ( path_handle -- size )
    FileSize,

    // ═══════════════════════════════════════════════════════════════════
    // Buffer Operations (for I/O data transfer)
    // ═══════════════════════════════════════════════════════════════════

    /// Create a new buffer with given capacity.
    /// Stack: ( capacity -- buffer_handle )
    BufferNew,

    /// Create a buffer from stack values (as bytes).
    /// Stack: ( byte1 byte2 ... byteN n -- buffer_handle )
    BufferFromStack,

    /// Push buffer contents to stack as byte values.
    /// Stack: ( buffer_handle -- byte1 byte2 ... byteN n )
    BufferToStack,

    /// Get buffer length.
    /// Stack: ( buffer_handle -- buffer_handle length )
    BufferLen,

    /// Read a byte from buffer at current position.
    /// Stack: ( buffer_handle -- buffer_handle byte )
    BufferReadByte,

    /// Write a byte to buffer at current position.
    /// Stack: ( buffer_handle byte -- buffer_handle )
    BufferWriteByte,

    /// Free a buffer.
    /// Stack: ( buffer_handle -- )
    BufferFree,

    // ═══════════════════════════════════════════════════════════════════
    // Network Operations
    // ═══════════════════════════════════════════════════════════════════

    /// Connect to a TCP server.
    /// Stack: ( host_handle port -- socket_handle )
    TcpConnect,

    /// Send data over a socket.
    /// Stack: ( socket_handle buffer_handle -- socket_handle bytes_sent )
    SocketSend,

    /// Receive data from a socket.
    /// Stack: ( socket_handle max_bytes -- socket_handle buffer_handle bytes_recv )
    SocketRecv,

    /// Close a socket.
    /// Stack: ( socket_handle -- )
    SocketClose,

    // ═══════════════════════════════════════════════════════════════════
    // Process Operations
    // ═══════════════════════════════════════════════════════════════════

    /// Execute a shell command and wait for completion.
    /// Stack: ( command_handle -- output_handle exit_code )
    ProcExec,

    // ═══════════════════════════════════════════════════════════════════
    // System Operations
    // ═══════════════════════════════════════════════════════════════════

    /// Get current time in milliseconds since Unix epoch.
    /// Stack: ( -- timestamp )
    Clock,

    /// Sleep for specified milliseconds.
    /// Stack: ( milliseconds -- )
    Sleep,

    /// Get a random number.
    /// Stack: ( -- random_value )
    Random,
}

impl OpCode {
    /// Get the name of this opcode as it appears in source code.
    pub fn name(&self) -> &'static str {
        match self {
            OpCode::Nop => "NOP",
            OpCode::Halt => "HALT",
            OpCode::Pop => "POP",
            OpCode::Dup => "DUP",
            OpCode::Swap => "SWAP",
            OpCode::Over => "OVER",
            OpCode::Rot => "ROT",
            OpCode::Depth => "DEPTH",
            OpCode::Pick => "PICK",
            OpCode::Add => "ADD",
            OpCode::Sub => "SUB",
            OpCode::Mul => "MUL",
            OpCode::Div => "DIV",
            OpCode::Mod => "MOD",
            OpCode::Neg => "NEG",
            OpCode::Abs => "ABS",
            OpCode::Min => "MIN",
            OpCode::Max => "MAX",
            OpCode::Sign => "SIGN",
            OpCode::Assert => "ASSERT",
            OpCode::Not => "NOT",
            OpCode::And => "AND",
            OpCode::Or => "OR",
            OpCode::Xor => "XOR",
            OpCode::Shl => "SHL",
            OpCode::Shr => "SHR",
            OpCode::Eq => "EQ",
            OpCode::Neq => "NEQ",
            OpCode::Lt => "LT",
            OpCode::Gt => "GT",
            OpCode::Lte => "LTE",
            OpCode::Gte => "GTE",
            OpCode::Slt => "SLT",
            OpCode::Sgt => "SGT",
            OpCode::Slte => "SLTE",
            OpCode::Sgte => "SGTE",
            OpCode::Oracle => "ORACLE",
            OpCode::Prophecy => "PROPHECY",
            OpCode::PresentRead => "PRESENT",
            OpCode::Paradox => "PARADOX",
            OpCode::Pack => "PACK",
            OpCode::Unpack => "UNPACK",
            OpCode::Index => "INDEX",
            OpCode::Store => "STORE",
            OpCode::Input => "INPUT",
            OpCode::Output => "OUTPUT",
            OpCode::Emit => "EMIT",
            OpCode::Roll => "ROLL",
            OpCode::Reverse => "REVERSE",
            OpCode::Exec => "EXEC",
            OpCode::Dip => "DIP",
            OpCode::Keep => "KEEP",
            OpCode::Bi => "BI",
            OpCode::Rec => "REC",
            OpCode::StrRev => "STR_REV",
            OpCode::StrCat => "STR_CAT",
            OpCode::StrSplit => "STR_SPLIT",
            // Vector operations
            OpCode::VecNew => "VEC_NEW",
            OpCode::VecPush => "VEC_PUSH",
            OpCode::VecPop => "VEC_POP",
            OpCode::VecGet => "VEC_GET",
            OpCode::VecSet => "VEC_SET",
            OpCode::VecLen => "VEC_LEN",
            // Hash table operations
            OpCode::HashNew => "HASH_NEW",
            OpCode::HashPut => "HASH_PUT",
            OpCode::HashGet => "HASH_GET",
            OpCode::HashDel => "HASH_DEL",
            OpCode::HashHas => "HASH_HAS",
            OpCode::HashLen => "HASH_LEN",
            // Set operations
            OpCode::SetNew => "SET_NEW",
            OpCode::SetAdd => "SET_ADD",
            OpCode::SetHas => "SET_HAS",
            OpCode::SetDel => "SET_DEL",
            OpCode::SetLen => "SET_LEN",
            // FFI operations
            OpCode::FFICall => "FFI_CALL",
            OpCode::FFICallNamed => "FFI_CALL_NAMED",
            // File I/O operations
            OpCode::FileOpen => "FILE_OPEN",
            OpCode::FileRead => "FILE_READ",
            OpCode::FileWrite => "FILE_WRITE",
            OpCode::FileSeek => "FILE_SEEK",
            OpCode::FileFlush => "FILE_FLUSH",
            OpCode::FileClose => "FILE_CLOSE",
            OpCode::FileExists => "FILE_EXISTS",
            OpCode::FileSize => "FILE_SIZE",
            // Buffer operations
            OpCode::BufferNew => "BUFFER_NEW",
            OpCode::BufferFromStack => "BUFFER_FROM_STACK",
            OpCode::BufferToStack => "BUFFER_TO_STACK",
            OpCode::BufferLen => "BUFFER_LEN",
            OpCode::BufferReadByte => "BUFFER_READ_BYTE",
            OpCode::BufferWriteByte => "BUFFER_WRITE_BYTE",
            OpCode::BufferFree => "BUFFER_FREE",
            // Network operations
            OpCode::TcpConnect => "TCP_CONNECT",
            OpCode::SocketSend => "SOCKET_SEND",
            OpCode::SocketRecv => "SOCKET_RECV",
            OpCode::SocketClose => "SOCKET_CLOSE",
            // Process operations
            OpCode::ProcExec => "PROC_EXEC",
            // System operations
            OpCode::Clock => "CLOCK",
            OpCode::Sleep => "SLEEP",
            OpCode::Random => "RANDOM",
        }
    }
    
    /// Get the stack effect: (inputs, outputs).
    pub fn stack_effect(&self) -> (usize, usize) {
        match self {
            OpCode::Nop | OpCode::Halt | OpCode::Paradox => (0, 0),
            OpCode::Pop | OpCode::Output | OpCode::Emit => (1, 0),
            OpCode::Dup => (1, 2),
            OpCode::Swap => (2, 2),
            OpCode::Over => (2, 3),

            OpCode::Rot => (3, 3),
            OpCode::Depth | OpCode::Input => (0, 1),
            OpCode::Pick => (1, 1),
            OpCode::Neg | OpCode::Abs | OpCode::Sign => (1, 1),
            OpCode::Assert => (2, 0), // Effectively consumes cond + len (and chars via len)
            OpCode::Not => (1, 1),
            OpCode::Add | OpCode::Sub | OpCode::Mul | OpCode::Div | OpCode::Mod |
            OpCode::Min | OpCode::Max |
            OpCode::And | OpCode::Or | OpCode::Xor | OpCode::Shl | OpCode::Shr |
            OpCode::Eq | OpCode::Neq | OpCode::Lt | OpCode::Gt | OpCode::Lte | OpCode::Gte |
            OpCode::Slt | OpCode::Sgt | OpCode::Slte | OpCode::Sgte => (2, 1),
            OpCode::Oracle | OpCode::PresentRead => (1, 1),
            OpCode::Prophecy => (2, 0),
            // Array opcodes have variable effects, these are minimums
            OpCode::Pack => (2, 0),    // base, n, (plus n values consumed)
            OpCode::Unpack => (2, 0),  // base, n (produces n values)
            OpCode::Index => (2, 1),   // base, index -> value
            OpCode::Store => (3, 0),   // value, base, index
            OpCode::Roll => (1, 0),    // pops n
            OpCode::Reverse => (1, 0), // pops n
            OpCode::Exec => (1, 0),    // pops quote_id
            OpCode::Dip => (2, 1),     // pops x, quote_id; pushes x (restores it)
            OpCode::Keep => (2, 1),    // pops x, quote; pushes x (under results?) -> complicated stack effect
            OpCode::Bi => (3, 0),      // pops x, p, q
            OpCode::Rec => (1, 0),     // pops quote (but pushes it back inside execution)
            OpCode::StrRev => (1, 1),  // pops len, pushes len
            OpCode::StrCat => (2, 1),  // pops len1, len2, pushes len_sum
            OpCode::StrSplit => (2, 1), // variable return
            // Vector operations
            OpCode::VecNew => (0, 1),   // ( -- vec )
            OpCode::VecPush => (2, 1),  // ( vec value -- vec )
            OpCode::VecPop => (1, 2),   // ( vec -- vec value )
            OpCode::VecGet => (2, 2),   // ( vec index -- vec value )
            OpCode::VecSet => (3, 1),   // ( vec value index -- vec )
            OpCode::VecLen => (1, 2),   // ( vec -- vec length )
            // Hash table operations
            OpCode::HashNew => (0, 1),  // ( -- hash )
            OpCode::HashPut => (3, 1),  // ( hash key value -- hash )
            OpCode::HashGet => (2, 3),  // ( hash key -- hash value found )
            OpCode::HashDel => (2, 1),  // ( hash key -- hash )
            OpCode::HashHas => (2, 2),  // ( hash key -- hash found )
            OpCode::HashLen => (1, 2),  // ( hash -- hash count )
            // Set operations
            OpCode::SetNew => (0, 1),   // ( -- set )
            OpCode::SetAdd => (2, 1),   // ( set value -- set )
            OpCode::SetHas => (2, 2),   // ( set value -- set found )
            OpCode::SetDel => (2, 1),   // ( set value -- set )
            OpCode::SetLen => (1, 2),   // ( set -- set count )
            // FFI operations (variable stack effects based on function)
            OpCode::FFICall => (1, 0),       // ( args... ffi_id -- results... ) - variable
            OpCode::FFICallNamed => (1, 0),  // ( args... name_handle -- results... ) - variable
            // File I/O operations
            OpCode::FileOpen => (2, 1),      // ( path_handle mode -- file_handle )
            OpCode::FileRead => (2, 3),      // ( file_handle max_bytes -- file_handle buffer_handle bytes_read )
            OpCode::FileWrite => (2, 2),     // ( file_handle buffer_handle -- file_handle bytes_written )
            OpCode::FileSeek => (3, 2),      // ( file_handle offset origin -- file_handle new_position )
            OpCode::FileFlush => (1, 1),     // ( file_handle -- file_handle )
            OpCode::FileClose => (1, 0),     // ( file_handle -- )
            OpCode::FileExists => (1, 1),    // ( path_handle -- exists )
            OpCode::FileSize => (1, 1),      // ( path_handle -- size )
            // Buffer operations
            OpCode::BufferNew => (1, 1),     // ( capacity -- buffer_handle )
            OpCode::BufferFromStack => (1, 1), // ( byte1..byteN n -- buffer_handle ) - variable input
            OpCode::BufferToStack => (1, 1), // ( buffer_handle -- byte1..byteN n ) - variable output
            OpCode::BufferLen => (1, 2),     // ( buffer_handle -- buffer_handle length )
            OpCode::BufferReadByte => (1, 2), // ( buffer_handle -- buffer_handle byte )
            OpCode::BufferWriteByte => (2, 1), // ( buffer_handle byte -- buffer_handle )
            OpCode::BufferFree => (1, 0),    // ( buffer_handle -- )
            // Network operations
            OpCode::TcpConnect => (2, 1),    // ( host_handle port -- socket_handle )
            OpCode::SocketSend => (2, 2),    // ( socket_handle buffer_handle -- socket_handle bytes_sent )
            OpCode::SocketRecv => (2, 3),    // ( socket_handle max_bytes -- socket_handle buffer_handle bytes_recv )
            OpCode::SocketClose => (1, 0),   // ( socket_handle -- )
            // Process operations
            OpCode::ProcExec => (1, 2),      // ( command_handle -- output_handle exit_code )
            // System operations
            OpCode::Clock => (0, 1),         // ( -- timestamp )
            OpCode::Sleep => (1, 0),         // ( milliseconds -- )
            OpCode::Random => (0, 1),        // ( -- random_value )
        }
    }
}

/// A statement in the OUROCHRONOS AST.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// Execute a single opcode.
    Op(OpCode),
    
    /// Push a constant value onto the stack.
    Push(Value),
    
    /// Structured IF statement.
    /// Pops the condition from stack; if non-zero, executes then_branch.
    If {
        then_branch: Vec<Stmt>,
        else_branch: Option<Vec<Stmt>>,
    },
    
    /// Structured WHILE loop.
    /// Evaluates cond block, pops result; if non-zero, executes body and repeats.
    While {
        cond: Vec<Stmt>,
        body: Vec<Stmt>,
    },
    
    /// A block of statements (scoped grouping).
    Block(Vec<Stmt>),
    
    /// Call a procedure by name.
    /// Stack effect depends on the procedure's parameter and return count.
    Call {
        name: String,
    },
    
    /// Pattern match on top-of-stack value.
    /// Pops value and selects matching case branch or default.
    Match {
        /// List of (pattern, body) pairs.
        cases: Vec<(u64, Vec<Stmt>)>,
        /// Default case if no pattern matches.
        default: Option<Vec<Stmt>>,
    },
    
    /// Temporal scope block.
    /// TEMPORAL <base> <size> { body }
    /// 
    /// Creates an isolated memory region for temporal operations.
    /// ORACLE/PROPHECY within the block are relative to base address.
    /// Changes within the scope are propagated to parent on successful exit.
    /// If the block induces a paradox, changes are discarded.
    TemporalScope {
        /// Base address for the isolated region.
        base: u64,
        /// Size of the isolated region.
        size: u64,
        /// Body of the temporal scope.
        body: Vec<Stmt>,
    },
}

/// Side effect behavior of a procedure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Effect {
    /// Procedure is pure (no side effects, deterministic given inputs).
    /// Implies no ORACLE reads and no PROPHECY writes.
    Pure,
    /// Procedure reads from the specific oracle address.
    Reads(u64),
    /// Procedure writes to the specific prophecy address.
    Writes(u64),
    /// Procedure performs temporal operations (ORACLE/PROPHECY).
    /// This is the general temporal effect that subsumes Reads/Writes.
    Temporal,
    /// Procedure performs I/O operations (INPUT/OUTPUT/EMIT).
    IO,
    /// Procedure may allocate or manipulate data structures.
    /// Used for VEC, HASH, SET operations.
    Alloc,
    /// Procedure performs FFI calls to external functions.
    /// FFI functions may have side effects not tracked by the type system.
    FFI,
    /// Procedure performs file I/O operations.
    FileIO,
    /// Procedure performs network operations.
    Network,
    /// Procedure performs process/system operations.
    System,
}

/// A procedure definition.
#[derive(Debug, Clone, PartialEq)]
pub struct Procedure {
    /// Procedure name.
    pub name: String,
    /// Parameter names (for documentation; all params come from stack).
    pub params: Vec<String>,
    /// Number of return values pushed to stack.
    pub returns: usize,
    /// Effect annotations declared by the user.
    pub effects: Vec<Effect>,
    /// Body of the procedure.
    pub body: Vec<Stmt>,
}

/// A complete OUROCHRONOS program.
#[derive(Debug, Clone)]
pub struct Program {
    /// Procedure definitions.
    pub procedures: Vec<Procedure>,
    /// Quotations (anonymous code blocks) indexed by ID.
    pub quotes: Vec<Vec<Stmt>>,
    /// Main program body.
    pub body: Vec<Stmt>,
    /// FFI declarations from FOREIGN blocks.
    pub ffi_declarations: Vec<crate::parser::FFIDeclaration>,
}

impl Program {
    /// Create an empty program.
    pub fn new() -> Self {
        Program {
            procedures: Vec::new(),
            quotes: Vec::new(),
            body: Vec::new(),
            ffi_declarations: Vec::new(),
        }
    }
    
    /// Check if the program is trivially consistent (no oracle operations).
    pub fn is_trivially_consistent(&self) -> bool {
        !self.contains_oracle(&self.body)
    }
    
    fn contains_oracle(&self, stmts: &[Stmt]) -> bool {
        for stmt in stmts {
            match stmt {
                Stmt::Op(OpCode::Oracle) => return true,
                Stmt::If { then_branch, else_branch } => {
                    if self.contains_oracle(then_branch) {
                        return true;
                    }
                    if let Some(else_stmts) = else_branch {
                        if self.contains_oracle(else_stmts) {
                            return true;
                        }
                    }
                }
                Stmt::While { cond, body } => {
                    if self.contains_oracle(cond) || self.contains_oracle(body) {
                        return true;
                    }
                }
                Stmt::Block(inner) => {
                    if self.contains_oracle(inner) {
                        return true;
                    }
                }
                Stmt::TemporalScope { body, .. } => {
                    if self.contains_oracle(body) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }
    
    /// Inline all procedure calls, replacing Stmt::Call with procedure bodies.
    /// Returns a new Program with all calls inlined.
    pub fn inline_procedures(&self) -> Self {
        let proc_map: std::collections::HashMap<String, &Procedure> = 
            self.procedures.iter().map(|p| (p.name.clone(), p)).collect();
        
        Program {
            procedures: Vec::new(), // Procedures are now inlined
            quotes: self.quotes.iter().map(|q| self.inline_stmts(q, &proc_map)).collect(),
            body: self.inline_stmts(&self.body, &proc_map),
            ffi_declarations: self.ffi_declarations.clone(),
        }
    }
    
    fn inline_stmts(&self, stmts: &[Stmt], procs: &std::collections::HashMap<String, &Procedure>) -> Vec<Stmt> {
        stmts.iter().map(|stmt| self.inline_stmt(stmt, procs)).collect()
    }
    
    fn inline_stmt(&self, stmt: &Stmt, procs: &std::collections::HashMap<String, &Procedure>) -> Stmt {
        match stmt {
            Stmt::Call { name } => {
                if let Some(proc) = procs.get(name) {
                    // Inline the procedure body
                    Stmt::Block(self.inline_stmts(&proc.body, procs))
                } else {
                    // Procedure not found - keep as is (will error at runtime)
                    stmt.clone()
                }
            }
            Stmt::If { then_branch, else_branch } => Stmt::If {
                then_branch: self.inline_stmts(then_branch, procs),
                else_branch: else_branch.as_ref().map(|e| self.inline_stmts(e, procs)),
            },
            Stmt::While { cond, body } => Stmt::While {
                cond: self.inline_stmts(cond, procs),
                body: self.inline_stmts(body, procs),
            },
            Stmt::Block(inner) => Stmt::Block(self.inline_stmts(inner, procs)),
            Stmt::TemporalScope { base, size, body } => Stmt::TemporalScope {
                base: *base,
                size: *size,
                body: self.inline_stmts(body, procs),
            },
            other => other.clone(),
        }
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}
