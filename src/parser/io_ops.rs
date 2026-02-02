//! I/O operations domain parser.
//!
//! Handles input/output, file, buffer, network, and system operations.

use crate::ast::{OpCode, Stmt};
use super::domain::{DomainParser, ParseContext};

/// Parser for I/O operations (file, buffer, network, system).
pub struct IoOpsParser;

impl IoOpsParser {
    pub const KEYWORDS: &'static [&'static str] = &[
        // Basic I/O
        "INPUT",
        // FFI operations
        "FFI_CALL", "FFI_CALL_NAMED",
        // File I/O operations
        "FILE_OPEN", "FILE_READ", "FILE_WRITE", "FILE_SEEK",
        "FILE_FLUSH", "FILE_CLOSE", "FILE_EXISTS", "FILE_SIZE",
        // Buffer operations
        "BUFFER_NEW", "BUFFER_FROM_STACK", "BUFFER_TO_STACK",
        "BUFFER_LEN", "BUFFER_READ_BYTE", "BUFFER_WRITE_BYTE", "BUFFER_FREE",
        // Network operations
        "TCP_CONNECT", "SOCKET_SEND", "SOCKET_RECV", "SOCKET_CLOSE",
        // Process operations
        "PROC_EXEC",
        // System operations
        "CLOCK", "SLEEP", "RANDOM",
    ];
}

impl DomainParser for IoOpsParser {
    fn keywords(&self) -> &'static [&'static str] {
        Self::KEYWORDS
    }

    fn parse<'a>(&self, keyword: &str, ctx: &mut ParseContext<'a>) -> Result<Stmt, String> {
        match keyword {
            // Basic I/O
            "INPUT" => ctx.emit_op(OpCode::Input),

            // FFI operations
            "FFI_CALL" => ctx.emit_op(OpCode::FFICall),
            "FFI_CALL_NAMED" => ctx.emit_op(OpCode::FFICallNamed),

            // File I/O operations
            "FILE_OPEN" => ctx.emit_op(OpCode::FileOpen),
            "FILE_READ" => ctx.emit_op(OpCode::FileRead),
            "FILE_WRITE" => ctx.emit_op(OpCode::FileWrite),
            "FILE_SEEK" => ctx.emit_op(OpCode::FileSeek),
            "FILE_FLUSH" => ctx.emit_op(OpCode::FileFlush),
            "FILE_CLOSE" => ctx.emit_op(OpCode::FileClose),
            "FILE_EXISTS" => ctx.emit_op(OpCode::FileExists),
            "FILE_SIZE" => ctx.emit_op(OpCode::FileSize),

            // Buffer operations
            "BUFFER_NEW" => ctx.emit_op(OpCode::BufferNew),
            "BUFFER_FROM_STACK" => ctx.emit_op(OpCode::BufferFromStack),
            "BUFFER_TO_STACK" => ctx.emit_op(OpCode::BufferToStack),
            "BUFFER_LEN" => ctx.emit_op(OpCode::BufferLen),
            "BUFFER_READ_BYTE" => ctx.emit_op(OpCode::BufferReadByte),
            "BUFFER_WRITE_BYTE" => ctx.emit_op(OpCode::BufferWriteByte),
            "BUFFER_FREE" => ctx.emit_op(OpCode::BufferFree),

            // Network operations
            "TCP_CONNECT" => ctx.emit_op(OpCode::TcpConnect),
            "SOCKET_SEND" => ctx.emit_op(OpCode::SocketSend),
            "SOCKET_RECV" => ctx.emit_op(OpCode::SocketRecv),
            "SOCKET_CLOSE" => ctx.emit_op(OpCode::SocketClose),

            // Process operations
            "PROC_EXEC" => ctx.emit_op(OpCode::ProcExec),

            // System operations
            "CLOCK" => ctx.emit_op(OpCode::Clock),
            "SLEEP" => ctx.emit_op(OpCode::Sleep),
            "RANDOM" => ctx.emit_op(OpCode::Random),

            _ => Err(format!("Unknown I/O operation: {}", keyword)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_io_keywords() {
        let parser = IoOpsParser;
        assert!(parser.keywords().contains(&"FILE_OPEN"));
        assert!(parser.keywords().contains(&"TCP_CONNECT"));
        assert!(parser.keywords().contains(&"CLOCK"));
    }

    #[test]
    fn test_parse_input() {
        let parser = IoOpsParser;
        let mut depth = 0;
        let mut ctx = ParseContext::new(&mut depth, super::super::domain::emit_op_helper);
        let result = parser.parse("INPUT", &mut ctx);
        assert!(result.is_ok());
        assert_eq!(depth, 1); // INPUT produces one value
    }
}
