//! I/O Operations for OUROCHRONOS.
//!
//! This module provides file, network, and process I/O capabilities while
//! maintaining temporal awareness. All I/O operations are treated as having
//! IO effects, meaning they cannot be called from pure contexts.
//!
//! # Design Principles
//!
//! 1. **Handle-Based**: All external resources use opaque handles
//! 2. **Effect Tracking**: All I/O operations have IO or Reads/Writes effects
//! 3. **Error Handling**: All operations return OuroResult with detailed errors
//! 4. **Temporal Awareness**: I/O is inherently non-deterministic; users must be aware
//!
//! # File Operations
//!
//! ```text
//! "data.txt" FILE_OPEN    # -- handle
//! 1024 FILE_READ          # handle bytes -- handle data_handle bytes_read
//! "output" FILE_WRITE     # handle data_handle len -- handle bytes_written
//! FILE_CLOSE              # handle --
//! ```
//!
//! # Network Operations
//!
//! ```text
//! "localhost" 8080 TCP_CONNECT  # host port -- socket
//! "GET /" SOCKET_SEND           # socket data len -- socket bytes_sent
//! 1024 SOCKET_RECV              # socket max_bytes -- socket data_handle bytes_recv
//! SOCKET_CLOSE                  # socket --
//! ```

use crate::core::{Value, Handle};
use crate::core::error::{OuroError, OuroResult, SourceLocation};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom, BufWriter};
use std::net::TcpStream;
use std::path::PathBuf;
use std::time::Duration;

// ═══════════════════════════════════════════════════════════════════════════════
// File Mode Flags
// ═══════════════════════════════════════════════════════════════════════════════

/// File open mode flags (can be combined with bitwise OR).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileMode(pub u8);

impl FileMode {
    /// Open for reading.
    pub const READ: FileMode = FileMode(0b0001);
    /// Open for writing.
    pub const WRITE: FileMode = FileMode(0b0010);
    /// Append to file.
    pub const APPEND: FileMode = FileMode(0b0100);
    /// Create file if it doesn't exist.
    pub const CREATE: FileMode = FileMode(0b1000);
    /// Truncate file on open.
    pub const TRUNCATE: FileMode = FileMode(0b10000);

    /// Check if mode includes reading.
    pub fn can_read(&self) -> bool {
        self.0 & Self::READ.0 != 0
    }

    /// Check if mode includes writing.
    pub fn can_write(&self) -> bool {
        self.0 & Self::WRITE.0 != 0 || self.0 & Self::APPEND.0 != 0
    }

    /// Check if mode includes creation.
    pub fn can_create(&self) -> bool {
        self.0 & Self::CREATE.0 != 0
    }

    /// Parse mode from u64 value.
    pub fn from_u64(val: u64) -> Self {
        FileMode(val as u8)
    }

    /// Convert mode flags to OpenOptions.
    pub fn to_open_options(&self) -> OpenOptions {
        let mut opts = OpenOptions::new();

        if self.can_read() || self.0 == 0 {
            opts.read(true);
        }
        if self.0 & Self::WRITE.0 != 0 {
            opts.write(true);
        }
        if self.0 & Self::APPEND.0 != 0 {
            opts.append(true);
        }
        if self.0 & Self::CREATE.0 != 0 {
            opts.create(true);
        }
        if self.0 & Self::TRUNCATE.0 != 0 {
            opts.truncate(true);
        }

        opts
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Seek Origin
// ═══════════════════════════════════════════════════════════════════════════════

/// Seek origin for file operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekOrigin {
    /// Seek from the beginning of the file.
    Start,
    /// Seek from the current position.
    Current,
    /// Seek from the end of the file.
    End,
}

impl SeekOrigin {
    /// Parse from u64 value.
    pub fn from_u64(val: u64) -> Self {
        match val {
            0 => SeekOrigin::Start,
            1 => SeekOrigin::Current,
            2 => SeekOrigin::End,
            _ => SeekOrigin::Start,
        }
    }

    /// Convert to std::io::SeekFrom.
    pub fn to_seek_from(&self, offset: i64) -> SeekFrom {
        match self {
            SeekOrigin::Start => SeekFrom::Start(offset as u64),
            SeekOrigin::Current => SeekFrom::Current(offset),
            SeekOrigin::End => SeekFrom::End(offset),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// File Handle
// ═══════════════════════════════════════════════════════════════════════════════

/// Wrapper around a file with metadata.
pub struct FileHandle {
    /// The underlying file.
    file: BufWriter<File>,
    /// Original path.
    path: PathBuf,
    /// Open mode.
    mode: FileMode,
    /// Current position (tracked for logging).
    position: u64,
}

impl FileHandle {
    /// Create a new file handle.
    pub fn new(file: File, path: PathBuf, mode: FileMode) -> Self {
        Self {
            file: BufWriter::new(file),
            path,
            mode,
            position: 0,
        }
    }

    /// Read up to `len` bytes into a buffer.
    pub fn read(&mut self, len: usize) -> OuroResult<Vec<u8>> {
        if !self.mode.can_read() && self.mode.0 != 0 {
            return Err(OuroError::IO {
                operation: "read".to_string(),
                path: Some(self.path.display().to_string()),
                message: "File not opened for reading".to_string(),
                location: SourceLocation::default(),
            });
        }

        let mut buffer = vec![0u8; len];
        let inner = self.file.get_mut();

        match inner.read(&mut buffer) {
            Ok(n) => {
                buffer.truncate(n);
                self.position += n as u64;
                Ok(buffer)
            }
            Err(e) => Err(OuroError::IO {
                operation: "read".to_string(),
                path: Some(self.path.display().to_string()),
                message: e.to_string(),
                location: SourceLocation::default(),
            }),
        }
    }

    /// Write bytes to the file.
    pub fn write(&mut self, data: &[u8]) -> OuroResult<usize> {
        if !self.mode.can_write() {
            return Err(OuroError::IO {
                operation: "write".to_string(),
                path: Some(self.path.display().to_string()),
                message: "File not opened for writing".to_string(),
                location: SourceLocation::default(),
            });
        }

        match self.file.write(data) {
            Ok(n) => {
                self.position += n as u64;
                Ok(n)
            }
            Err(e) => Err(OuroError::IO {
                operation: "write".to_string(),
                path: Some(self.path.display().to_string()),
                message: e.to_string(),
                location: SourceLocation::default(),
            }),
        }
    }

    /// Seek to a position in the file.
    pub fn seek(&mut self, origin: SeekOrigin, offset: i64) -> OuroResult<u64> {
        let inner = self.file.get_mut();
        match inner.seek(origin.to_seek_from(offset)) {
            Ok(pos) => {
                self.position = pos;
                Ok(pos)
            }
            Err(e) => Err(OuroError::IO {
                operation: "seek".to_string(),
                path: Some(self.path.display().to_string()),
                message: e.to_string(),
                location: SourceLocation::default(),
            }),
        }
    }

    /// Flush the write buffer.
    pub fn flush(&mut self) -> OuroResult<()> {
        self.file.flush().map_err(|e| OuroError::IO {
            operation: "flush".to_string(),
            path: Some(self.path.display().to_string()),
            message: e.to_string(),
            location: SourceLocation::default(),
        })
    }

    /// Get the current position.
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Get the file path.
    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Socket Handle
// ═══════════════════════════════════════════════════════════════════════════════

/// Wrapper around a TCP socket.
pub struct SocketHandle {
    /// The underlying stream.
    stream: TcpStream,
    /// Remote address.
    remote_addr: String,
    /// Read timeout in milliseconds.
    read_timeout: Option<Duration>,
    /// Write timeout in milliseconds.
    write_timeout: Option<Duration>,
}

impl SocketHandle {
    /// Create a new socket handle from a TcpStream.
    pub fn new(stream: TcpStream, remote_addr: String) -> Self {
        Self {
            stream,
            remote_addr,
            read_timeout: None,
            write_timeout: None,
        }
    }

    /// Connect to a remote address.
    pub fn connect(addr: &str) -> OuroResult<Self> {
        let stream = TcpStream::connect(addr).map_err(|e| OuroError::IO {
            operation: "tcp_connect".to_string(),
            path: Some(addr.to_string()),
            message: e.to_string(),
            location: SourceLocation::default(),
        })?;

        Ok(Self::new(stream, addr.to_string()))
    }

    /// Set read timeout.
    pub fn set_read_timeout(&mut self, timeout_ms: u64) -> OuroResult<()> {
        let duration = if timeout_ms == 0 {
            None
        } else {
            Some(Duration::from_millis(timeout_ms))
        };

        self.stream.set_read_timeout(duration).map_err(|e| OuroError::IO {
            operation: "set_read_timeout".to_string(),
            path: Some(self.remote_addr.clone()),
            message: e.to_string(),
            location: SourceLocation::default(),
        })?;

        self.read_timeout = duration;
        Ok(())
    }

    /// Set write timeout.
    pub fn set_write_timeout(&mut self, timeout_ms: u64) -> OuroResult<()> {
        let duration = if timeout_ms == 0 {
            None
        } else {
            Some(Duration::from_millis(timeout_ms))
        };

        self.stream.set_write_timeout(duration).map_err(|e| OuroError::IO {
            operation: "set_write_timeout".to_string(),
            path: Some(self.remote_addr.clone()),
            message: e.to_string(),
            location: SourceLocation::default(),
        })?;

        self.write_timeout = duration;
        Ok(())
    }

    /// Send data over the socket.
    pub fn send(&mut self, data: &[u8]) -> OuroResult<usize> {
        self.stream.write(data).map_err(|e| OuroError::IO {
            operation: "socket_send".to_string(),
            path: Some(self.remote_addr.clone()),
            message: e.to_string(),
            location: SourceLocation::default(),
        })
    }

    /// Receive data from the socket.
    pub fn recv(&mut self, max_len: usize) -> OuroResult<Vec<u8>> {
        let mut buffer = vec![0u8; max_len];
        let n = self.stream.read(&mut buffer).map_err(|e| OuroError::IO {
            operation: "socket_recv".to_string(),
            path: Some(self.remote_addr.clone()),
            message: e.to_string(),
            location: SourceLocation::default(),
        })?;

        buffer.truncate(n);
        Ok(buffer)
    }

    /// Get the remote address.
    pub fn remote_addr(&self) -> &str {
        &self.remote_addr
    }

    /// Shutdown the socket.
    pub fn shutdown(&self) -> OuroResult<()> {
        self.stream.shutdown(std::net::Shutdown::Both).map_err(|e| OuroError::IO {
            operation: "socket_shutdown".to_string(),
            path: Some(self.remote_addr.clone()),
            message: e.to_string(),
            location: SourceLocation::default(),
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Buffer Handle
// ═══════════════════════════════════════════════════════════════════════════════

/// A byte buffer for I/O operations.
#[derive(Debug, Clone)]
pub struct Buffer {
    /// The data.
    data: Vec<u8>,
    /// Read/write position.
    position: usize,
}

impl Buffer {
    /// Create a new empty buffer.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            position: 0,
        }
    }

    /// Create a buffer with a specific capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            position: 0,
        }
    }

    /// Create a buffer from existing data.
    pub fn from_data(data: Vec<u8>) -> Self {
        Self { data, position: 0 }
    }

    /// Get the length of the buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the current position.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Set the position.
    pub fn set_position(&mut self, pos: usize) {
        self.position = pos.min(self.data.len());
    }

    /// Read a single byte.
    pub fn read_byte(&mut self) -> Option<u8> {
        if self.position < self.data.len() {
            let byte = self.data[self.position];
            self.position += 1;
            Some(byte)
        } else {
            None
        }
    }

    /// Read up to `len` bytes.
    pub fn read(&mut self, len: usize) -> Vec<u8> {
        let end = (self.position + len).min(self.data.len());
        let result = self.data[self.position..end].to_vec();
        self.position = end;
        result
    }

    /// Write bytes to the buffer.
    pub fn write(&mut self, data: &[u8]) {
        // Extend buffer if writing past end
        if self.position + data.len() > self.data.len() {
            self.data.resize(self.position + data.len(), 0);
        }

        self.data[self.position..self.position + data.len()].copy_from_slice(data);
        self.position += data.len();
    }

    /// Append bytes to the end of the buffer.
    pub fn append(&mut self, data: &[u8]) {
        self.data.extend_from_slice(data);
    }

    /// Get a reference to the underlying data.
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get a mutable reference to the underlying data.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.data.clear();
        self.position = 0;
    }

    /// Convert to a string (lossy).
    pub fn to_string_lossy(&self) -> String {
        String::from_utf8_lossy(&self.data).to_string()
    }
}

impl Default for Buffer {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// I/O Context
// ═══════════════════════════════════════════════════════════════════════════════

/// Context for all I/O operations.
///
/// This manages file handles, sockets, and buffers for the VM.
pub struct IOContext {
    /// Open file handles.
    files: HashMap<Handle, FileHandle>,
    /// Open socket handles.
    sockets: HashMap<Handle, SocketHandle>,
    /// Buffers for data transfer.
    buffers: HashMap<Handle, Buffer>,
    /// Next handle ID.
    next_handle: Handle,
    /// Current working directory.
    cwd: PathBuf,
    /// Maximum open files.
    max_open_files: usize,
    /// Maximum open sockets.
    max_open_sockets: usize,
    /// I/O operations log (for temporal debugging).
    io_log: Vec<IOLogEntry>,
}

/// Log entry for I/O operations (used for temporal debugging).
#[derive(Debug, Clone)]
pub struct IOLogEntry {
    /// Operation name.
    pub operation: String,
    /// Handle involved.
    pub handle: Option<Handle>,
    /// Path or address.
    pub path: Option<String>,
    /// Bytes transferred.
    pub bytes: Option<usize>,
    /// Success status.
    pub success: bool,
    /// Timestamp (monotonic counter).
    pub timestamp: u64,
}

impl IOContext {
    /// Create a new I/O context.
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
            sockets: HashMap::new(),
            buffers: HashMap::new(),
            next_handle: 1, // 0 is reserved for null/invalid
            cwd: std::env::current_dir().unwrap_or_default(),
            max_open_files: 256,
            max_open_sockets: 64,
            io_log: Vec::new(),
        }
    }

    /// Generate a new handle ID.
    fn new_handle(&mut self) -> Handle {
        let handle = self.next_handle;
        self.next_handle += 1;
        handle
    }

    /// Log an I/O operation.
    fn log(&mut self, operation: &str, handle: Option<Handle>, path: Option<String>, bytes: Option<usize>, success: bool) {
        self.io_log.push(IOLogEntry {
            operation: operation.to_string(),
            handle,
            path,
            bytes,
            success,
            timestamp: self.io_log.len() as u64,
        });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // File Operations
    // ═══════════════════════════════════════════════════════════════════════════

    /// Open a file.
    pub fn file_open(&mut self, path: &str, mode: FileMode) -> OuroResult<Handle> {
        if self.files.len() >= self.max_open_files {
            return Err(OuroError::IO {
                operation: "file_open".to_string(),
                path: Some(path.to_string()),
                message: "Too many open files".to_string(),
                location: SourceLocation::default(),
            });
        }

        let full_path = if path.starts_with('/') {
            PathBuf::from(path)
        } else {
            self.cwd.join(path)
        };

        let file = mode.to_open_options()
            .open(&full_path)
            .map_err(|e| OuroError::IO {
                operation: "file_open".to_string(),
                path: Some(full_path.display().to_string()),
                message: e.to_string(),
                location: SourceLocation::default(),
            })?;

        let handle = self.new_handle();
        self.files.insert(handle, FileHandle::new(file, full_path.clone(), mode));
        self.log("file_open", Some(handle), Some(full_path.display().to_string()), None, true);

        Ok(handle)
    }

    /// Read from a file.
    pub fn file_read(&mut self, handle: Handle, len: usize) -> OuroResult<(Handle, usize)> {
        let file = self.files.get_mut(&handle).ok_or_else(|| OuroError::IO {
            operation: "file_read".to_string(),
            path: None,
            message: format!("Invalid file handle: {}", handle),
            location: SourceLocation::default(),
        })?;

        let data = file.read(len)?;
        let bytes_read = data.len();
        let path = file.path().display().to_string();

        // Create a buffer for the data
        let buffer_handle = self.new_handle();
        self.buffers.insert(buffer_handle, Buffer::from_data(data));
        self.log("file_read", Some(handle), Some(path), Some(bytes_read), true);

        Ok((buffer_handle, bytes_read))
    }

    /// Write to a file.
    pub fn file_write(&mut self, handle: Handle, buffer_handle: Handle) -> OuroResult<usize> {
        let buffer = self.buffers.get(&buffer_handle).ok_or_else(|| OuroError::IO {
            operation: "file_write".to_string(),
            path: None,
            message: format!("Invalid buffer handle: {}", buffer_handle),
            location: SourceLocation::default(),
        })?;

        let data = buffer.as_slice().to_vec();

        let file = self.files.get_mut(&handle).ok_or_else(|| OuroError::IO {
            operation: "file_write".to_string(),
            path: None,
            message: format!("Invalid file handle: {}", handle),
            location: SourceLocation::default(),
        })?;

        let path = file.path().display().to_string();
        let bytes_written = file.write(&data)?;
        self.log("file_write", Some(handle), Some(path), Some(bytes_written), true);

        Ok(bytes_written)
    }

    /// Seek in a file.
    pub fn file_seek(&mut self, handle: Handle, origin: SeekOrigin, offset: i64) -> OuroResult<u64> {
        let file = self.files.get_mut(&handle).ok_or_else(|| OuroError::IO {
            operation: "file_seek".to_string(),
            path: None,
            message: format!("Invalid file handle: {}", handle),
            location: SourceLocation::default(),
        })?;

        let path = file.path().display().to_string();
        let pos = file.seek(origin, offset)?;
        self.log("file_seek", Some(handle), Some(path), None, true);

        Ok(pos)
    }

    /// Flush a file.
    pub fn file_flush(&mut self, handle: Handle) -> OuroResult<()> {
        let file = self.files.get_mut(&handle).ok_or_else(|| OuroError::IO {
            operation: "file_flush".to_string(),
            path: None,
            message: format!("Invalid file handle: {}", handle),
            location: SourceLocation::default(),
        })?;

        file.flush()
    }

    /// Close a file.
    pub fn file_close(&mut self, handle: Handle) -> OuroResult<()> {
        if self.files.remove(&handle).is_some() {
            self.log("file_close", Some(handle), None, None, true);
            Ok(())
        } else {
            Err(OuroError::IO {
                operation: "file_close".to_string(),
                path: None,
                message: format!("Invalid file handle: {}", handle),
                location: SourceLocation::default(),
            })
        }
    }

    /// Check if a file exists.
    pub fn file_exists(&self, path: &str) -> bool {
        let full_path = if path.starts_with('/') {
            PathBuf::from(path)
        } else {
            self.cwd.join(path)
        };

        full_path.exists()
    }

    /// Get file size.
    pub fn file_size(&self, path: &str) -> OuroResult<u64> {
        let full_path = if path.starts_with('/') {
            PathBuf::from(path)
        } else {
            self.cwd.join(path)
        };

        std::fs::metadata(&full_path)
            .map(|m| m.len())
            .map_err(|e| OuroError::IO {
                operation: "file_size".to_string(),
                path: Some(full_path.display().to_string()),
                message: e.to_string(),
                location: SourceLocation::default(),
            })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Socket Operations
    // ═══════════════════════════════════════════════════════════════════════════

    /// Connect to a TCP server.
    pub fn tcp_connect(&mut self, host: &str, port: u16) -> OuroResult<Handle> {
        if self.sockets.len() >= self.max_open_sockets {
            return Err(OuroError::IO {
                operation: "tcp_connect".to_string(),
                path: Some(format!("{}:{}", host, port)),
                message: "Too many open sockets".to_string(),
                location: SourceLocation::default(),
            });
        }

        let addr = format!("{}:{}", host, port);
        let socket = SocketHandle::connect(&addr)?;

        let handle = self.new_handle();
        self.sockets.insert(handle, socket);
        self.log("tcp_connect", Some(handle), Some(addr), None, true);

        Ok(handle)
    }

    /// Send data over a socket.
    pub fn socket_send(&mut self, handle: Handle, buffer_handle: Handle) -> OuroResult<usize> {
        let buffer = self.buffers.get(&buffer_handle).ok_or_else(|| OuroError::IO {
            operation: "socket_send".to_string(),
            path: None,
            message: format!("Invalid buffer handle: {}", buffer_handle),
            location: SourceLocation::default(),
        })?;

        let data = buffer.as_slice().to_vec();

        let socket = self.sockets.get_mut(&handle).ok_or_else(|| OuroError::IO {
            operation: "socket_send".to_string(),
            path: None,
            message: format!("Invalid socket handle: {}", handle),
            location: SourceLocation::default(),
        })?;

        let addr = socket.remote_addr().to_string();
        let bytes_sent = socket.send(&data)?;
        self.log("socket_send", Some(handle), Some(addr), Some(bytes_sent), true);

        Ok(bytes_sent)
    }

    /// Receive data from a socket.
    pub fn socket_recv(&mut self, handle: Handle, max_len: usize) -> OuroResult<(Handle, usize)> {
        let socket = self.sockets.get_mut(&handle).ok_or_else(|| OuroError::IO {
            operation: "socket_recv".to_string(),
            path: None,
            message: format!("Invalid socket handle: {}", handle),
            location: SourceLocation::default(),
        })?;

        let addr = socket.remote_addr().to_string();
        let data = socket.recv(max_len)?;
        let bytes_recv = data.len();

        let buffer_handle = self.new_handle();
        self.buffers.insert(buffer_handle, Buffer::from_data(data));
        self.log("socket_recv", Some(handle), Some(addr), Some(bytes_recv), true);

        Ok((buffer_handle, bytes_recv))
    }

    /// Close a socket.
    pub fn socket_close(&mut self, handle: Handle) -> OuroResult<()> {
        if let Some(socket) = self.sockets.remove(&handle) {
            socket.shutdown().ok(); // Ignore shutdown errors
            self.log("socket_close", Some(handle), None, None, true);
            Ok(())
        } else {
            Err(OuroError::IO {
                operation: "socket_close".to_string(),
                path: None,
                message: format!("Invalid socket handle: {}", handle),
                location: SourceLocation::default(),
            })
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Buffer Operations
    // ═══════════════════════════════════════════════════════════════════════════

    /// Create a new buffer.
    pub fn buffer_new(&mut self, capacity: usize) -> Handle {
        let handle = self.new_handle();
        self.buffers.insert(handle, Buffer::with_capacity(capacity));
        handle
    }

    /// Create a buffer from string data.
    pub fn buffer_from_string(&mut self, s: &str) -> Handle {
        let handle = self.new_handle();
        self.buffers.insert(handle, Buffer::from_data(s.as_bytes().to_vec()));
        handle
    }

    /// Create a buffer from memory values.
    pub fn buffer_from_values(&mut self, values: &[Value]) -> Handle {
        let handle = self.new_handle();
        let data: Vec<u8> = values.iter().map(|v| v.val as u8).collect();
        self.buffers.insert(handle, Buffer::from_data(data));
        handle
    }

    /// Get buffer length.
    pub fn buffer_len(&self, handle: Handle) -> OuroResult<usize> {
        self.buffers.get(&handle)
            .map(|b| b.len())
            .ok_or_else(|| OuroError::IO {
                operation: "buffer_len".to_string(),
                path: None,
                message: format!("Invalid buffer handle: {}", handle),
                location: SourceLocation::default(),
            })
    }

    /// Read a byte from a buffer.
    pub fn buffer_read_byte(&mut self, handle: Handle) -> OuroResult<Option<u8>> {
        self.buffers.get_mut(&handle)
            .map(|b| b.read_byte())
            .ok_or_else(|| OuroError::IO {
                operation: "buffer_read_byte".to_string(),
                path: None,
                message: format!("Invalid buffer handle: {}", handle),
                location: SourceLocation::default(),
            })
    }

    /// Write a byte to a buffer.
    pub fn buffer_write_byte(&mut self, handle: Handle, byte: u8) -> OuroResult<()> {
        self.buffers.get_mut(&handle)
            .map(|b| b.write(&[byte]))
            .ok_or_else(|| OuroError::IO {
                operation: "buffer_write_byte".to_string(),
                path: None,
                message: format!("Invalid buffer handle: {}", handle),
                location: SourceLocation::default(),
            })
    }

    /// Get buffer as string.
    pub fn buffer_to_string(&self, handle: Handle) -> OuroResult<String> {
        self.buffers.get(&handle)
            .map(|b| b.to_string_lossy())
            .ok_or_else(|| OuroError::IO {
                operation: "buffer_to_string".to_string(),
                path: None,
                message: format!("Invalid buffer handle: {}", handle),
                location: SourceLocation::default(),
            })
    }

    /// Copy buffer contents to stack as values.
    pub fn buffer_to_values(&self, handle: Handle) -> OuroResult<Vec<Value>> {
        self.buffers.get(&handle)
            .map(|b| b.as_slice().iter().map(|&byte| Value::new(byte as u64)).collect())
            .ok_or_else(|| OuroError::IO {
                operation: "buffer_to_values".to_string(),
                path: None,
                message: format!("Invalid buffer handle: {}", handle),
                location: SourceLocation::default(),
            })
    }

    /// Free a buffer.
    pub fn buffer_free(&mut self, handle: Handle) -> bool {
        self.buffers.remove(&handle).is_some()
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Process Operations
    // ═══════════════════════════════════════════════════════════════════════════

    /// Execute a command and capture output.
    pub fn exec(&mut self, command: &str) -> OuroResult<(Handle, i32)> {
        use std::process::Command;

        let output = if cfg!(windows) {
            Command::new("cmd")
                .args(["/C", command])
                .output()
        } else {
            Command::new("sh")
                .args(["-c", command])
                .output()
        };

        match output {
            Ok(output) => {
                let mut data = output.stdout;
                data.extend_from_slice(&output.stderr);

                let buffer_handle = self.new_handle();
                self.buffers.insert(buffer_handle, Buffer::from_data(data));

                let exit_code = output.status.code().unwrap_or(-1);
                self.log("exec", Some(buffer_handle), Some(command.to_string()), None, true);

                Ok((buffer_handle, exit_code))
            }
            Err(e) => Err(OuroError::IO {
                operation: "exec".to_string(),
                path: Some(command.to_string()),
                message: e.to_string(),
                location: SourceLocation::default(),
            }),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Utility Methods
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get the I/O operation log.
    pub fn get_log(&self) -> &[IOLogEntry] {
        &self.io_log
    }

    /// Clear the I/O operation log.
    pub fn clear_log(&mut self) {
        self.io_log.clear();
    }

    /// Get the current working directory.
    pub fn get_cwd(&self) -> &PathBuf {
        &self.cwd
    }

    /// Set the current working directory.
    pub fn set_cwd(&mut self, path: &str) -> OuroResult<()> {
        let new_cwd = PathBuf::from(path);
        if new_cwd.is_dir() {
            self.cwd = new_cwd;
            Ok(())
        } else {
            Err(OuroError::IO {
                operation: "set_cwd".to_string(),
                path: Some(path.to_string()),
                message: "Not a directory".to_string(),
                location: SourceLocation::default(),
            })
        }
    }

    /// Close all open handles.
    pub fn close_all(&mut self) {
        // Close all files
        self.files.clear();

        // Close all sockets
        for (_, socket) in self.sockets.drain() {
            socket.shutdown().ok();
        }

        // Clear buffers
        self.buffers.clear();

        self.log("close_all", None, None, None, true);
    }

    /// Get statistics about open resources.
    pub fn stats(&self) -> IOStats {
        IOStats {
            open_files: self.files.len(),
            open_sockets: self.sockets.len(),
            active_buffers: self.buffers.len(),
            total_operations: self.io_log.len(),
        }
    }
}

impl Default for IOContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for IOContext {
    fn drop(&mut self) {
        self.close_all();
    }
}

/// Statistics about I/O resources.
#[derive(Debug, Clone)]
pub struct IOStats {
    /// Number of open files.
    pub open_files: usize,
    /// Number of open sockets.
    pub open_sockets: usize,
    /// Number of active buffers.
    pub active_buffers: usize,
    /// Total I/O operations performed.
    pub total_operations: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_file_mode_flags() {
        let read = FileMode::READ;
        assert!(read.can_read());
        assert!(!read.can_write());

        let write = FileMode::WRITE;
        assert!(write.can_write());

        let both = FileMode(FileMode::READ.0 | FileMode::WRITE.0);
        assert!(both.can_read());
        assert!(both.can_write());
    }

    #[test]
    fn test_seek_origin() {
        assert_eq!(SeekOrigin::from_u64(0), SeekOrigin::Start);
        assert_eq!(SeekOrigin::from_u64(1), SeekOrigin::Current);
        assert_eq!(SeekOrigin::from_u64(2), SeekOrigin::End);
        assert_eq!(SeekOrigin::from_u64(99), SeekOrigin::Start);
    }

    #[test]
    fn test_buffer_operations() {
        let mut buffer = Buffer::new();
        assert!(buffer.is_empty());

        buffer.write(b"hello");
        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.position(), 5);

        buffer.set_position(0);
        let data = buffer.read(5);
        assert_eq!(data, b"hello");
    }

    #[test]
    fn test_buffer_from_data() {
        let buffer = Buffer::from_data(b"test data".to_vec());
        assert_eq!(buffer.len(), 9);
        assert_eq!(buffer.to_string_lossy(), "test data");
    }

    #[test]
    fn test_io_context_buffer() {
        let mut ctx = IOContext::new();

        let handle = ctx.buffer_new(1024);
        assert!(handle > 0);

        let len = ctx.buffer_len(handle).unwrap();
        assert_eq!(len, 0); // Empty buffer

        ctx.buffer_write_byte(handle, b'A').unwrap();
        let len = ctx.buffer_len(handle).unwrap();
        assert_eq!(len, 1);
    }

    #[test]
    fn test_io_context_buffer_from_string() {
        let mut ctx = IOContext::new();

        let handle = ctx.buffer_from_string("Hello, World!");
        let s = ctx.buffer_to_string(handle).unwrap();
        assert_eq!(s, "Hello, World!");
    }

    #[test]
    fn test_io_stats() {
        let ctx = IOContext::new();
        let stats = ctx.stats();

        assert_eq!(stats.open_files, 0);
        assert_eq!(stats.open_sockets, 0);
        assert_eq!(stats.active_buffers, 0);
    }

    #[test]
    fn test_file_exists() {
        let ctx = IOContext::new();

        // Test with current directory (should exist)
        assert!(ctx.file_exists("."));

        // Test with non-existent file
        assert!(!ctx.file_exists("this_file_definitely_does_not_exist_12345.txt"));
    }

    #[test]
    fn test_io_log() {
        let mut ctx = IOContext::new();

        let handle = ctx.buffer_new(100);
        ctx.buffer_free(handle);

        // Operations are logged
        // (buffer_new doesn't log, but close_all does)
        ctx.close_all();

        let log = ctx.get_log();
        assert!(!log.is_empty());
    }
}
