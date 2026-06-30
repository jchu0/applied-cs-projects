//! HTTP/1.1 protocol implementation.
//!
//! Implements request/response parsing, chunked transfer encoding,
//! and keep-alive connections.

use crate::{Error, Result};
use bytes::{BufMut, BytesMut};
use std::collections::HashMap;
use std::str;

/// HTTP version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpVersion {
    Http10,
    Http11,
}

impl HttpVersion {
    /// Parse HTTP version from string.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "HTTP/1.0" => Ok(HttpVersion::Http10),
            "HTTP/1.1" => Ok(HttpVersion::Http11),
            _ => Err(Error::Parse(format!("Unknown HTTP version: {}", s))),
        }
    }

    /// Convert to string.
    pub fn as_str(&self) -> &'static str {
        match self {
            HttpVersion::Http10 => "HTTP/1.0",
            HttpVersion::Http11 => "HTTP/1.1",
        }
    }
}

/// HTTP method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
    Put,
    Delete,
    Head,
    Options,
    Patch,
    Connect,
    Trace,
}

impl Method {
    /// Parse HTTP method from string.
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_uppercase().as_str() {
            "GET" => Ok(Method::Get),
            "POST" => Ok(Method::Post),
            "PUT" => Ok(Method::Put),
            "DELETE" => Ok(Method::Delete),
            "HEAD" => Ok(Method::Head),
            "OPTIONS" => Ok(Method::Options),
            "PATCH" => Ok(Method::Patch),
            "CONNECT" => Ok(Method::Connect),
            "TRACE" => Ok(Method::Trace),
            _ => Err(Error::Parse(format!("Unknown HTTP method: {}", s))),
        }
    }

    /// Convert to string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Method::Get => "GET",
            Method::Post => "POST",
            Method::Put => "PUT",
            Method::Delete => "DELETE",
            Method::Head => "HEAD",
            Method::Options => "OPTIONS",
            Method::Patch => "PATCH",
            Method::Connect => "CONNECT",
            Method::Trace => "TRACE",
        }
    }
}

/// HTTP status code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatusCode(pub u16);

impl StatusCode {
    // 2xx Success
    pub const OK: StatusCode = StatusCode(200);
    pub const CREATED: StatusCode = StatusCode(201);
    pub const ACCEPTED: StatusCode = StatusCode(202);
    pub const NO_CONTENT: StatusCode = StatusCode(204);

    // 3xx Redirection
    pub const MOVED_PERMANENTLY: StatusCode = StatusCode(301);
    pub const FOUND: StatusCode = StatusCode(302);
    pub const NOT_MODIFIED: StatusCode = StatusCode(304);

    // 4xx Client Errors
    pub const BAD_REQUEST: StatusCode = StatusCode(400);
    pub const UNAUTHORIZED: StatusCode = StatusCode(401);
    pub const FORBIDDEN: StatusCode = StatusCode(403);
    pub const NOT_FOUND: StatusCode = StatusCode(404);
    pub const METHOD_NOT_ALLOWED: StatusCode = StatusCode(405);
    pub const CONFLICT: StatusCode = StatusCode(409);
    pub const GONE: StatusCode = StatusCode(410);
    pub const LENGTH_REQUIRED: StatusCode = StatusCode(411);
    pub const PAYLOAD_TOO_LARGE: StatusCode = StatusCode(413);
    pub const URI_TOO_LONG: StatusCode = StatusCode(414);
    pub const UNSUPPORTED_MEDIA_TYPE: StatusCode = StatusCode(415);
    pub const TOO_MANY_REQUESTS: StatusCode = StatusCode(429);

    // 5xx Server Errors
    pub const INTERNAL_SERVER_ERROR: StatusCode = StatusCode(500);
    pub const NOT_IMPLEMENTED: StatusCode = StatusCode(501);
    pub const BAD_GATEWAY: StatusCode = StatusCode(502);
    pub const SERVICE_UNAVAILABLE: StatusCode = StatusCode(503);
    pub const GATEWAY_TIMEOUT: StatusCode = StatusCode(504);

    /// Get reason phrase.
    pub fn reason_phrase(&self) -> &'static str {
        match self.0 {
            200 => "OK",
            201 => "Created",
            202 => "Accepted",
            204 => "No Content",
            301 => "Moved Permanently",
            302 => "Found",
            304 => "Not Modified",
            400 => "Bad Request",
            401 => "Unauthorized",
            403 => "Forbidden",
            404 => "Not Found",
            405 => "Method Not Allowed",
            409 => "Conflict",
            410 => "Gone",
            411 => "Length Required",
            413 => "Payload Too Large",
            414 => "URI Too Long",
            415 => "Unsupported Media Type",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            501 => "Not Implemented",
            502 => "Bad Gateway",
            503 => "Service Unavailable",
            504 => "Gateway Timeout",
            _ => "Unknown",
        }
    }

    /// Check if status indicates success.
    pub fn is_success(&self) -> bool {
        self.0 >= 200 && self.0 < 300
    }

    /// Check if status indicates redirect.
    pub fn is_redirect(&self) -> bool {
        self.0 >= 300 && self.0 < 400
    }

    /// Check if status indicates client error.
    pub fn is_client_error(&self) -> bool {
        self.0 >= 400 && self.0 < 500
    }

    /// Check if status indicates server error.
    pub fn is_server_error(&self) -> bool {
        self.0 >= 500 && self.0 < 600
    }
}

/// HTTP headers.
#[derive(Debug, Clone, Default)]
pub struct Headers {
    /// Header map (case-insensitive keys).
    headers: HashMap<String, Vec<String>>,
}

impl Headers {
    /// Create empty headers.
    pub fn new() -> Self {
        Self {
            headers: HashMap::new(),
        }
    }

    /// Get header value.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_lowercase())
            .and_then(|v| v.first())
            .map(|s| s.as_str())
    }

    /// Get all values for a header.
    pub fn get_all(&self, name: &str) -> Option<&Vec<String>> {
        self.headers.get(&name.to_lowercase())
    }

    /// Set header value.
    pub fn set(&mut self, name: impl Into<String>, value: impl Into<String>) {
        let key = name.into().to_lowercase();
        self.headers.insert(key, vec![value.into()]);
    }

    /// Append header value.
    pub fn append(&mut self, name: impl Into<String>, value: impl Into<String>) {
        let key = name.into().to_lowercase();
        self.headers
            .entry(key)
            .or_default()
            .push(value.into());
    }

    /// Remove header.
    pub fn remove(&mut self, name: &str) {
        self.headers.remove(&name.to_lowercase());
    }

    /// Check if header exists.
    pub fn contains(&self, name: &str) -> bool {
        self.headers.contains_key(&name.to_lowercase())
    }

    /// Get Content-Length.
    pub fn content_length(&self) -> Option<usize> {
        self.get("content-length")
            .and_then(|v| v.parse().ok())
    }

    /// Get Content-Type.
    pub fn content_type(&self) -> Option<&str> {
        self.get("content-type")
    }

    /// Check if chunked transfer encoding.
    pub fn is_chunked(&self) -> bool {
        self.get("transfer-encoding")
            .map(|v| v.to_lowercase().contains("chunked"))
            .unwrap_or(false)
    }

    /// Check if keep-alive.
    pub fn is_keep_alive(&self, version: HttpVersion) -> bool {
        match self.get("connection") {
            Some(conn) => conn.to_lowercase() == "keep-alive",
            None => version == HttpVersion::Http11,
        }
    }

    /// Iterate over headers.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.headers
            .iter()
            .flat_map(|(k, v)| v.iter().map(move |val| (k.as_str(), val.as_str())))
    }

    /// Serialize headers.
    pub fn serialize(&self) -> String {
        let mut result = String::new();
        for (name, values) in &self.headers {
            for value in values {
                result.push_str(name);
                result.push_str(": ");
                result.push_str(value);
                result.push_str("\r\n");
            }
        }
        result
    }
}

/// HTTP request.
#[derive(Debug, Clone)]
pub struct Request {
    /// HTTP method.
    pub method: Method,
    /// Request URI.
    pub uri: String,
    /// HTTP version.
    pub version: HttpVersion,
    /// Headers.
    pub headers: Headers,
    /// Body.
    pub body: BytesMut,
}

impl Request {
    /// Create a new request.
    pub fn new(method: Method, uri: impl Into<String>) -> Self {
        Self {
            method,
            uri: uri.into(),
            version: HttpVersion::Http11,
            headers: Headers::new(),
            body: BytesMut::new(),
        }
    }

    /// Set header.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.set(name, value);
        self
    }

    /// Set body.
    pub fn body(mut self, body: impl Into<BytesMut>) -> Self {
        self.body = body.into();
        self
    }

    /// Serialize request.
    pub fn serialize(&self) -> BytesMut {
        let mut buf = BytesMut::new();

        // Request line
        buf.put_slice(self.method.as_str().as_bytes());
        buf.put_slice(b" ");
        buf.put_slice(self.uri.as_bytes());
        buf.put_slice(b" ");
        buf.put_slice(self.version.as_str().as_bytes());
        buf.put_slice(b"\r\n");

        // Headers
        buf.put_slice(self.headers.serialize().as_bytes());

        // Content-Length if body present
        if !self.body.is_empty() && !self.headers.contains("content-length") {
            buf.put_slice(b"Content-Length: ");
            buf.put_slice(self.body.len().to_string().as_bytes());
            buf.put_slice(b"\r\n");
        }

        buf.put_slice(b"\r\n");

        // Body
        if !self.body.is_empty() {
            buf.extend_from_slice(&self.body);
        }

        buf
    }
}

/// HTTP response.
#[derive(Debug, Clone)]
pub struct Response {
    /// HTTP version.
    pub version: HttpVersion,
    /// Status code.
    pub status: StatusCode,
    /// Headers.
    pub headers: Headers,
    /// Body.
    pub body: BytesMut,
}

impl Response {
    /// Create a new response.
    pub fn new(status: StatusCode) -> Self {
        Self {
            version: HttpVersion::Http11,
            status,
            headers: Headers::new(),
            body: BytesMut::new(),
        }
    }

    /// Set header.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.set(name, value);
        self
    }

    /// Set body.
    pub fn body(mut self, body: impl Into<BytesMut>) -> Self {
        self.body = body.into();
        self
    }

    /// Serialize response.
    pub fn serialize(&self) -> BytesMut {
        let mut buf = BytesMut::new();

        // Status line
        buf.put_slice(self.version.as_str().as_bytes());
        buf.put_slice(b" ");
        buf.put_slice(self.status.0.to_string().as_bytes());
        buf.put_slice(b" ");
        buf.put_slice(self.status.reason_phrase().as_bytes());
        buf.put_slice(b"\r\n");

        // Headers
        buf.put_slice(self.headers.serialize().as_bytes());

        // Content-Length if body present
        if !self.body.is_empty() && !self.headers.contains("content-length") {
            buf.put_slice(b"Content-Length: ");
            buf.put_slice(self.body.len().to_string().as_bytes());
            buf.put_slice(b"\r\n");
        }

        buf.put_slice(b"\r\n");

        // Body
        if !self.body.is_empty() {
            buf.extend_from_slice(&self.body);
        }

        buf
    }
}

/// HTTP request parser.
pub struct RequestParser {
    /// Parsing state.
    state: ParserState,
    /// Accumulated data.
    buffer: BytesMut,
    /// Parsed request (in progress).
    request: Option<Request>,
    /// Remaining body bytes.
    body_remaining: usize,
    /// Chunked decoder state.
    chunk_state: ChunkState,
}

/// Parser state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParserState {
    RequestLine,
    Headers,
    Body,
    Chunked,
    Complete,
}

/// Chunked transfer encoding state.
#[derive(Debug, Clone, Copy)]
enum ChunkState {
    Size,
    Data { remaining: usize },
    DataEnd,
    Trailer,
}

impl RequestParser {
    /// Create a new request parser.
    pub fn new() -> Self {
        Self {
            state: ParserState::RequestLine,
            buffer: BytesMut::new(),
            request: None,
            body_remaining: 0,
            chunk_state: ChunkState::Size,
        }
    }

    /// Feed data to the parser.
    pub fn feed(&mut self, data: &[u8]) -> Result<Option<Request>> {
        self.buffer.extend_from_slice(data);
        self.parse()
    }

    /// Parse accumulated data.
    fn parse(&mut self) -> Result<Option<Request>> {
        loop {
            match self.state {
                ParserState::RequestLine => {
                    if !self.parse_request_line()? {
                        return Ok(None);
                    }
                }
                ParserState::Headers => {
                    if !self.parse_headers()? {
                        return Ok(None);
                    }
                }
                ParserState::Body => {
                    if !self.parse_body()? {
                        return Ok(None);
                    }
                }
                ParserState::Chunked => {
                    if !self.parse_chunked()? {
                        return Ok(None);
                    }
                }
                ParserState::Complete => {
                    let request = self.request.take();
                    self.reset();
                    return Ok(request);
                }
            }
        }
    }

    /// Parse request line.
    fn parse_request_line(&mut self) -> Result<bool> {
        if let Some(line_end) = self.find_line_end() {
            let line = str::from_utf8(&self.buffer[..line_end])
                .map_err(|_| Error::Parse("Invalid UTF-8 in request line".into()))?;

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 3 {
                return Err(Error::Parse("Invalid request line".into()));
            }

            let method = Method::parse(parts[0])?;
            let uri = parts[1].to_string();
            let version = HttpVersion::parse(parts[2])?;

            self.request = Some(Request {
                method,
                uri,
                version,
                headers: Headers::new(),
                body: BytesMut::new(),
            });

            // Remove parsed data
            let _ = self.buffer.split_to(line_end + 2);
            self.state = ParserState::Headers;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Parse headers.
    fn parse_headers(&mut self) -> Result<bool> {
        loop {
            if let Some(line_end) = self.find_line_end() {
                if line_end == 0 {
                    // Empty line - end of headers
                    let _ = self.buffer.split_to(2);

                    let request = self.request.as_ref().unwrap();
                    if request.headers.is_chunked() {
                        self.state = ParserState::Chunked;
                    } else if let Some(len) = request.headers.content_length() {
                        self.body_remaining = len;
                        self.state = ParserState::Body;
                    } else {
                        self.state = ParserState::Complete;
                    }
                    return Ok(true);
                }

                let line = str::from_utf8(&self.buffer[..line_end])
                    .map_err(|_| Error::Parse("Invalid UTF-8 in header".into()))?;

                if let Some(colon) = line.find(':') {
                    let name = line[..colon].trim();
                    let value = line[colon + 1..].trim();

                    if let Some(request) = &mut self.request {
                        request.headers.append(name, value);
                    }
                }

                let _ = self.buffer.split_to(line_end + 2);
            } else {
                return Ok(false);
            }
        }
    }

    /// Parse body.
    fn parse_body(&mut self) -> Result<bool> {
        if self.buffer.len() >= self.body_remaining {
            let body_data = self.buffer.split_to(self.body_remaining);
            if let Some(request) = &mut self.request {
                request.body = body_data;
            }
            self.state = ParserState::Complete;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Parse chunked transfer encoding.
    fn parse_chunked(&mut self) -> Result<bool> {
        loop {
            match self.chunk_state {
                ChunkState::Size => {
                    if let Some(line_end) = self.find_line_end() {
                        let line = str::from_utf8(&self.buffer[..line_end])
                            .map_err(|_| Error::Parse("Invalid chunk size".into()))?;

                        // Parse hex size (ignore extensions)
                        let size_str = line.split(';').next().unwrap_or(line).trim();
                        let size = usize::from_str_radix(size_str, 16)
                            .map_err(|_| Error::Parse("Invalid chunk size".into()))?;

                        let _ = self.buffer.split_to(line_end + 2);

                        if size == 0 {
                            self.chunk_state = ChunkState::Trailer;
                        } else {
                            self.chunk_state = ChunkState::Data { remaining: size };
                        }
                    } else {
                        return Ok(false);
                    }
                }

                ChunkState::Data { remaining } => {
                    if self.buffer.len() >= remaining {
                        let chunk_data = self.buffer.split_to(remaining);
                        if let Some(request) = &mut self.request {
                            request.body.extend_from_slice(&chunk_data);
                        }
                        self.chunk_state = ChunkState::DataEnd;
                    } else {
                        return Ok(false);
                    }
                }

                ChunkState::DataEnd => {
                    if self.buffer.len() >= 2 {
                        let _ = self.buffer.split_to(2); // CRLF
                        self.chunk_state = ChunkState::Size;
                    } else {
                        return Ok(false);
                    }
                }

                ChunkState::Trailer => {
                    // Skip trailer headers
                    if let Some(line_end) = self.find_line_end() {
                        if line_end == 0 {
                            let _ = self.buffer.split_to(2);
                            self.state = ParserState::Complete;
                            return Ok(true);
                        }
                        let _ = self.buffer.split_to(line_end + 2);
                    } else {
                        return Ok(false);
                    }
                }
            }
        }
    }

    /// Find line ending (CRLF).
    fn find_line_end(&self) -> Option<usize> {
        for i in 0..self.buffer.len().saturating_sub(1) {
            if self.buffer[i] == b'\r' && self.buffer[i + 1] == b'\n' {
                return Some(i);
            }
        }
        None
    }

    /// Reset parser state.
    fn reset(&mut self) {
        self.state = ParserState::RequestLine;
        self.request = None;
        self.body_remaining = 0;
        self.chunk_state = ChunkState::Size;
    }
}

impl Default for RequestParser {
    fn default() -> Self {
        Self::new()
    }
}

/// HTTP response parser.
pub struct ResponseParser {
    /// Parsing state.
    state: ParserState,
    /// Accumulated data.
    buffer: BytesMut,
    /// Parsed response (in progress).
    response: Option<Response>,
    /// Remaining body bytes.
    body_remaining: usize,
    /// Chunked decoder state.
    chunk_state: ChunkState,
}

impl ResponseParser {
    /// Create a new response parser.
    pub fn new() -> Self {
        Self {
            state: ParserState::RequestLine, // Reusing for status line
            buffer: BytesMut::new(),
            response: None,
            body_remaining: 0,
            chunk_state: ChunkState::Size,
        }
    }

    /// Feed data to the parser.
    pub fn feed(&mut self, data: &[u8]) -> Result<Option<Response>> {
        self.buffer.extend_from_slice(data);
        self.parse()
    }

    /// Parse accumulated data.
    fn parse(&mut self) -> Result<Option<Response>> {
        loop {
            match self.state {
                ParserState::RequestLine => {
                    if !self.parse_status_line()? {
                        return Ok(None);
                    }
                }
                ParserState::Headers => {
                    if !self.parse_headers()? {
                        return Ok(None);
                    }
                }
                ParserState::Body => {
                    if !self.parse_body()? {
                        return Ok(None);
                    }
                }
                ParserState::Chunked => {
                    if !self.parse_chunked()? {
                        return Ok(None);
                    }
                }
                ParserState::Complete => {
                    let response = self.response.take();
                    self.reset();
                    return Ok(response);
                }
            }
        }
    }

    /// Parse status line.
    fn parse_status_line(&mut self) -> Result<bool> {
        if let Some(line_end) = self.find_line_end() {
            let line = str::from_utf8(&self.buffer[..line_end])
                .map_err(|_| Error::Parse("Invalid UTF-8 in status line".into()))?;

            let mut parts = line.splitn(3, ' ');

            let version = HttpVersion::parse(
                parts.next().ok_or_else(|| Error::Parse("Missing version".into()))?
            )?;

            let status_code: u16 = parts
                .next()
                .ok_or_else(|| Error::Parse("Missing status code".into()))?
                .parse()
                .map_err(|_| Error::Parse("Invalid status code".into()))?;

            // Reason phrase is optional
            let _ = parts.next();

            self.response = Some(Response {
                version,
                status: StatusCode(status_code),
                headers: Headers::new(),
                body: BytesMut::new(),
            });

            let _ = self.buffer.split_to(line_end + 2);
            self.state = ParserState::Headers;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Parse headers.
    fn parse_headers(&mut self) -> Result<bool> {
        loop {
            if let Some(line_end) = self.find_line_end() {
                if line_end == 0 {
                    let _ = self.buffer.split_to(2);

                    let response = self.response.as_ref().unwrap();
                    if response.headers.is_chunked() {
                        self.state = ParserState::Chunked;
                    } else if let Some(len) = response.headers.content_length() {
                        self.body_remaining = len;
                        self.state = ParserState::Body;
                    } else {
                        self.state = ParserState::Complete;
                    }
                    return Ok(true);
                }

                let line = str::from_utf8(&self.buffer[..line_end])
                    .map_err(|_| Error::Parse("Invalid UTF-8 in header".into()))?;

                if let Some(colon) = line.find(':') {
                    let name = line[..colon].trim();
                    let value = line[colon + 1..].trim();

                    if let Some(response) = &mut self.response {
                        response.headers.append(name, value);
                    }
                }

                let _ = self.buffer.split_to(line_end + 2);
            } else {
                return Ok(false);
            }
        }
    }

    /// Parse body.
    fn parse_body(&mut self) -> Result<bool> {
        if self.buffer.len() >= self.body_remaining {
            let body_data = self.buffer.split_to(self.body_remaining);
            if let Some(response) = &mut self.response {
                response.body = body_data;
            }
            self.state = ParserState::Complete;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Parse chunked transfer encoding.
    fn parse_chunked(&mut self) -> Result<bool> {
        loop {
            match self.chunk_state {
                ChunkState::Size => {
                    if let Some(line_end) = self.find_line_end() {
                        let line = str::from_utf8(&self.buffer[..line_end])
                            .map_err(|_| Error::Parse("Invalid chunk size".into()))?;

                        let size_str = line.split(';').next().unwrap_or(line).trim();
                        let size = usize::from_str_radix(size_str, 16)
                            .map_err(|_| Error::Parse("Invalid chunk size".into()))?;

                        let _ = self.buffer.split_to(line_end + 2);

                        if size == 0 {
                            self.chunk_state = ChunkState::Trailer;
                        } else {
                            self.chunk_state = ChunkState::Data { remaining: size };
                        }
                    } else {
                        return Ok(false);
                    }
                }

                ChunkState::Data { remaining } => {
                    if self.buffer.len() >= remaining {
                        let chunk_data = self.buffer.split_to(remaining);
                        if let Some(response) = &mut self.response {
                            response.body.extend_from_slice(&chunk_data);
                        }
                        self.chunk_state = ChunkState::DataEnd;
                    } else {
                        return Ok(false);
                    }
                }

                ChunkState::DataEnd => {
                    if self.buffer.len() >= 2 {
                        let _ = self.buffer.split_to(2);
                        self.chunk_state = ChunkState::Size;
                    } else {
                        return Ok(false);
                    }
                }

                ChunkState::Trailer => {
                    if let Some(line_end) = self.find_line_end() {
                        if line_end == 0 {
                            let _ = self.buffer.split_to(2);
                            self.state = ParserState::Complete;
                            return Ok(true);
                        }
                        let _ = self.buffer.split_to(line_end + 2);
                    } else {
                        return Ok(false);
                    }
                }
            }
        }
    }

    /// Find line ending.
    fn find_line_end(&self) -> Option<usize> {
        for i in 0..self.buffer.len().saturating_sub(1) {
            if self.buffer[i] == b'\r' && self.buffer[i + 1] == b'\n' {
                return Some(i);
            }
        }
        None
    }

    /// Reset parser.
    fn reset(&mut self) {
        self.state = ParserState::RequestLine;
        self.response = None;
        self.body_remaining = 0;
        self.chunk_state = ChunkState::Size;
    }
}

impl Default for ResponseParser {
    fn default() -> Self {
        Self::new()
    }
}

/// URL parser.
#[derive(Debug, Clone)]
pub struct Url {
    /// Scheme (http/https).
    pub scheme: String,
    /// Host.
    pub host: String,
    /// Port.
    pub port: Option<u16>,
    /// Path.
    pub path: String,
    /// Query string.
    pub query: Option<String>,
}

impl Url {
    /// Parse URL from string.
    pub fn parse(url: &str) -> Result<Self> {
        let (scheme, rest) = url
            .split_once("://")
            .ok_or_else(|| Error::Parse("Missing scheme".into()))?;

        let (authority, path_and_query) = rest
            .split_once('/')
            .map(|(a, p)| (a, format!("/{}", p)))
            .unwrap_or((rest, "/".into()));

        let (host, port) = if let Some((h, p)) = authority.split_once(':') {
            let port: u16 = p.parse().map_err(|_| Error::Parse("Invalid port".into()))?;
            (h.to_string(), Some(port))
        } else {
            (authority.to_string(), None)
        };

        let (path, query) = if let Some((p, q)) = path_and_query.split_once('?') {
            (p.to_string(), Some(q.to_string()))
        } else {
            (path_and_query, None)
        };

        Ok(Self {
            scheme: scheme.to_string(),
            host,
            port,
            path,
            query,
        })
    }

    /// Get effective port.
    pub fn effective_port(&self) -> u16 {
        self.port.unwrap_or_else(|| {
            if self.scheme == "https" {
                443
            } else {
                80
            }
        })
    }

    /// Get host with port.
    pub fn host_port(&self) -> String {
        if let Some(port) = self.port {
            format!("{}:{}", self.host, port)
        } else {
            self.host.clone()
        }
    }

    /// Get full path with query.
    pub fn path_and_query(&self) -> String {
        if let Some(q) = &self.query {
            format!("{}?{}", self.path, q)
        } else {
            self.path.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_parse() {
        let mut parser = RequestParser::new();

        let data = b"GET /index.html HTTP/1.1\r\n\
                     Host: example.com\r\n\
                     Content-Length: 5\r\n\
                     \r\n\
                     hello";

        let request = parser.feed(data).unwrap().unwrap();
        assert_eq!(request.method, Method::Get);
        assert_eq!(request.uri, "/index.html");
        assert_eq!(request.headers.get("host"), Some("example.com"));
        assert_eq!(&request.body[..], b"hello");
    }

    #[test]
    fn test_chunked_request() {
        let mut parser = RequestParser::new();

        let data = b"POST /upload HTTP/1.1\r\n\
                     Transfer-Encoding: chunked\r\n\
                     \r\n\
                     5\r\n\
                     hello\r\n\
                     6\r\n\
                     world!\r\n\
                     0\r\n\
                     \r\n";

        let request = parser.feed(data).unwrap().unwrap();
        assert_eq!(&request.body[..], b"helloworld!");
    }

    #[test]
    fn test_response_serialize() {
        let response = Response::new(StatusCode::OK)
            .header("Content-Type", "text/plain")
            .body(BytesMut::from(&b"Hello"[..]));

        let serialized = response.serialize();
        let expected = "HTTP/1.1 200 OK\r\n\
                        content-type: text/plain\r\n\
                        Content-Length: 5\r\n\
                        \r\n\
                        Hello";

        // Check key parts
        assert!(serialized.starts_with(b"HTTP/1.1 200 OK"));
        assert!(serialized.ends_with(b"Hello"));
    }

    #[test]
    fn test_url_parse() {
        let url = Url::parse("https://example.com:8080/path?query=1").unwrap();
        assert_eq!(url.scheme, "https");
        assert_eq!(url.host, "example.com");
        assert_eq!(url.port, Some(8080));
        assert_eq!(url.path, "/path");
        assert_eq!(url.query, Some("query=1".into()));
    }

    #[test]
    fn test_headers() {
        let mut headers = Headers::new();
        headers.set("Content-Type", "application/json");
        headers.append("Accept", "text/html");
        headers.append("Accept", "application/xml");

        assert_eq!(headers.get("content-type"), Some("application/json"));
        assert_eq!(headers.get_all("accept").unwrap().len(), 2);
    }
}
