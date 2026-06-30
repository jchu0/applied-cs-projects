//! Comprehensive HTTP/1.1 parser and handler tests
//!
//! Tests for request parsing, response parsing, chunked encoding,
//! headers, URL parsing, and HTTP builders.

use network_stack::http::*;
use bytes::BytesMut;

// =============================================================================
// HTTP Version Tests
// =============================================================================

#[cfg(test)]
mod http_version_tests {
    use super::*;

    #[test]
    fn test_parse_http10() {
        let version = HttpVersion::parse("HTTP/1.0").unwrap();
        assert_eq!(version, HttpVersion::Http10);
    }

    #[test]
    fn test_parse_http11() {
        let version = HttpVersion::parse("HTTP/1.1").unwrap();
        assert_eq!(version, HttpVersion::Http11);
    }

    #[test]
    fn test_parse_unknown_version() {
        let result = HttpVersion::parse("HTTP/2.0");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_version() {
        let result = HttpVersion::parse("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_version_as_str() {
        assert_eq!(HttpVersion::Http10.as_str(), "HTTP/1.0");
        assert_eq!(HttpVersion::Http11.as_str(), "HTTP/1.1");
    }
}

// =============================================================================
// HTTP Method Tests
// =============================================================================

#[cfg(test)]
mod http_method_tests {
    use super::*;

    #[test]
    fn test_parse_all_methods() {
        assert_eq!(Method::parse("GET").unwrap(), Method::Get);
        assert_eq!(Method::parse("POST").unwrap(), Method::Post);
        assert_eq!(Method::parse("PUT").unwrap(), Method::Put);
        assert_eq!(Method::parse("DELETE").unwrap(), Method::Delete);
        assert_eq!(Method::parse("HEAD").unwrap(), Method::Head);
        assert_eq!(Method::parse("OPTIONS").unwrap(), Method::Options);
        assert_eq!(Method::parse("PATCH").unwrap(), Method::Patch);
        assert_eq!(Method::parse("CONNECT").unwrap(), Method::Connect);
        assert_eq!(Method::parse("TRACE").unwrap(), Method::Trace);
    }

    #[test]
    fn test_parse_case_insensitive() {
        assert_eq!(Method::parse("get").unwrap(), Method::Get);
        assert_eq!(Method::parse("Get").unwrap(), Method::Get);
        assert_eq!(Method::parse("gEt").unwrap(), Method::Get);
    }

    #[test]
    fn test_parse_unknown_method() {
        let result = Method::parse("UNKNOWN");
        assert!(result.is_err());
    }

    #[test]
    fn test_method_as_str() {
        assert_eq!(Method::Get.as_str(), "GET");
        assert_eq!(Method::Post.as_str(), "POST");
        assert_eq!(Method::Put.as_str(), "PUT");
        assert_eq!(Method::Delete.as_str(), "DELETE");
        assert_eq!(Method::Head.as_str(), "HEAD");
        assert_eq!(Method::Options.as_str(), "OPTIONS");
        assert_eq!(Method::Patch.as_str(), "PATCH");
        assert_eq!(Method::Connect.as_str(), "CONNECT");
        assert_eq!(Method::Trace.as_str(), "TRACE");
    }
}

// =============================================================================
// HTTP Status Code Tests
// =============================================================================

#[cfg(test)]
mod http_status_code_tests {
    use super::*;

    #[test]
    fn test_status_code_constants() {
        assert_eq!(StatusCode::OK.0, 200);
        assert_eq!(StatusCode::CREATED.0, 201);
        assert_eq!(StatusCode::NO_CONTENT.0, 204);
        assert_eq!(StatusCode::BAD_REQUEST.0, 400);
        assert_eq!(StatusCode::NOT_FOUND.0, 404);
        assert_eq!(StatusCode::INTERNAL_SERVER_ERROR.0, 500);
    }

    #[test]
    fn test_reason_phrases() {
        assert_eq!(StatusCode::OK.reason_phrase(), "OK");
        assert_eq!(StatusCode::CREATED.reason_phrase(), "Created");
        assert_eq!(StatusCode::NOT_FOUND.reason_phrase(), "Not Found");
        assert_eq!(StatusCode::INTERNAL_SERVER_ERROR.reason_phrase(), "Internal Server Error");
    }

    #[test]
    fn test_is_success() {
        assert!(StatusCode::OK.is_success());
        assert!(StatusCode::CREATED.is_success());
        assert!(StatusCode::NO_CONTENT.is_success());
        assert!(!StatusCode::BAD_REQUEST.is_success());
        assert!(!StatusCode::NOT_FOUND.is_success());
        assert!(!StatusCode::INTERNAL_SERVER_ERROR.is_success());
    }

    #[test]
    fn test_is_redirect() {
        assert!(StatusCode::MOVED_PERMANENTLY.is_redirect());
        assert!(StatusCode::FOUND.is_redirect());
        assert!(StatusCode::NOT_MODIFIED.is_redirect());
        assert!(!StatusCode::OK.is_redirect());
        assert!(!StatusCode::NOT_FOUND.is_redirect());
    }

    #[test]
    fn test_is_client_error() {
        assert!(StatusCode::BAD_REQUEST.is_client_error());
        assert!(StatusCode::UNAUTHORIZED.is_client_error());
        assert!(StatusCode::NOT_FOUND.is_client_error());
        assert!(!StatusCode::OK.is_client_error());
        assert!(!StatusCode::INTERNAL_SERVER_ERROR.is_client_error());
    }

    #[test]
    fn test_is_server_error() {
        assert!(StatusCode::INTERNAL_SERVER_ERROR.is_server_error());
        assert!(StatusCode::BAD_GATEWAY.is_server_error());
        assert!(StatusCode::SERVICE_UNAVAILABLE.is_server_error());
        assert!(!StatusCode::OK.is_server_error());
        assert!(!StatusCode::NOT_FOUND.is_server_error());
    }

    #[test]
    fn test_custom_status_code() {
        let custom = StatusCode(418); // I'm a teapot
        assert_eq!(custom.0, 418);
        assert_eq!(custom.reason_phrase(), "Unknown");
        assert!(custom.is_client_error());
    }
}

// =============================================================================
// HTTP Headers Tests
// =============================================================================

#[cfg(test)]
mod http_headers_tests {
    use super::*;

    #[test]
    fn test_headers_new() {
        let headers = Headers::new();
        assert!(headers.get("any").is_none());
    }

    #[test]
    fn test_headers_set_get() {
        let mut headers = Headers::new();
        headers.set("Content-Type", "application/json");

        assert_eq!(headers.get("Content-Type"), Some("application/json"));
    }

    #[test]
    fn test_headers_case_insensitive() {
        let mut headers = Headers::new();
        headers.set("Content-Type", "application/json");

        assert_eq!(headers.get("content-type"), Some("application/json"));
        assert_eq!(headers.get("CONTENT-TYPE"), Some("application/json"));
        assert_eq!(headers.get("Content-type"), Some("application/json"));
    }

    #[test]
    fn test_headers_append() {
        let mut headers = Headers::new();
        headers.append("Accept", "text/html");
        headers.append("Accept", "application/json");

        let values = headers.get_all("Accept").unwrap();
        assert_eq!(values.len(), 2);
        assert!(values.contains(&"text/html".to_string()));
        assert!(values.contains(&"application/json".to_string()));
    }

    #[test]
    fn test_headers_set_overwrites() {
        let mut headers = Headers::new();
        headers.set("Content-Type", "text/plain");
        headers.set("Content-Type", "application/json");

        assert_eq!(headers.get("Content-Type"), Some("application/json"));
    }

    #[test]
    fn test_headers_remove() {
        let mut headers = Headers::new();
        headers.set("Content-Type", "application/json");
        headers.remove("Content-Type");

        assert!(headers.get("Content-Type").is_none());
    }

    #[test]
    fn test_headers_contains() {
        let mut headers = Headers::new();
        headers.set("Content-Type", "application/json");

        assert!(headers.contains("Content-Type"));
        assert!(headers.contains("content-type"));
        assert!(!headers.contains("Accept"));
    }

    #[test]
    fn test_content_length() {
        let mut headers = Headers::new();
        headers.set("Content-Length", "42");

        assert_eq!(headers.content_length(), Some(42));
    }

    #[test]
    fn test_content_length_invalid() {
        let mut headers = Headers::new();
        headers.set("Content-Length", "not-a-number");

        assert_eq!(headers.content_length(), None);
    }

    #[test]
    fn test_content_type() {
        let mut headers = Headers::new();
        headers.set("Content-Type", "text/html; charset=utf-8");

        assert_eq!(headers.content_type(), Some("text/html; charset=utf-8"));
    }

    #[test]
    fn test_is_chunked() {
        let mut headers = Headers::new();
        headers.set("Transfer-Encoding", "chunked");

        assert!(headers.is_chunked());

        headers.set("Transfer-Encoding", "gzip, chunked");
        assert!(headers.is_chunked());

        headers.set("Transfer-Encoding", "gzip");
        assert!(!headers.is_chunked());
    }

    #[test]
    fn test_is_keep_alive_http11_default() {
        let headers = Headers::new();
        // HTTP/1.1 defaults to keep-alive
        assert!(headers.is_keep_alive(HttpVersion::Http11));
    }

    #[test]
    fn test_is_keep_alive_http10_default() {
        let headers = Headers::new();
        // HTTP/1.0 defaults to close
        assert!(!headers.is_keep_alive(HttpVersion::Http10));
    }

    #[test]
    fn test_is_keep_alive_explicit() {
        let mut headers = Headers::new();
        headers.set("Connection", "keep-alive");

        assert!(headers.is_keep_alive(HttpVersion::Http10));
        assert!(headers.is_keep_alive(HttpVersion::Http11));
    }

    #[test]
    fn test_is_keep_alive_close() {
        let mut headers = Headers::new();
        headers.set("Connection", "close");

        assert!(!headers.is_keep_alive(HttpVersion::Http10));
        assert!(!headers.is_keep_alive(HttpVersion::Http11));
    }

    #[test]
    fn test_headers_serialize() {
        let mut headers = Headers::new();
        headers.set("Content-Type", "text/plain");
        headers.set("X-Custom", "value");

        let serialized = headers.serialize();

        assert!(serialized.contains("content-type: text/plain\r\n"));
        assert!(serialized.contains("x-custom: value\r\n"));
    }

    #[test]
    fn test_headers_iter() {
        let mut headers = Headers::new();
        headers.set("Content-Type", "text/plain");
        headers.set("Accept", "application/json");

        let pairs: Vec<_> = headers.iter().collect();
        assert_eq!(pairs.len(), 2);
    }
}

// =============================================================================
// HTTP Request Tests
// =============================================================================

#[cfg(test)]
mod http_request_tests {
    use super::*;

    #[test]
    fn test_request_new() {
        let request = Request::new(Method::Get, "/index.html");

        assert_eq!(request.method, Method::Get);
        assert_eq!(request.uri, "/index.html");
        assert_eq!(request.version, HttpVersion::Http11);
        assert!(request.body.is_empty());
    }

    #[test]
    fn test_request_builder_pattern() {
        let request = Request::new(Method::Post, "/api/users")
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .body(BytesMut::from(&b"{\"name\":\"test\"}"[..]));

        assert_eq!(request.method, Method::Post);
        assert_eq!(request.uri, "/api/users");
        assert_eq!(request.headers.get("Content-Type"), Some("application/json"));
        assert_eq!(&request.body[..], b"{\"name\":\"test\"}");
    }

    #[test]
    fn test_request_serialize() {
        let request = Request::new(Method::Get, "/path")
            .header("Host", "example.com");

        let serialized = request.serialize();
        let serialized_str = String::from_utf8_lossy(&serialized);

        assert!(serialized_str.starts_with("GET /path HTTP/1.1\r\n"));
        assert!(serialized_str.contains("host: example.com\r\n"));
        assert!(serialized_str.ends_with("\r\n\r\n"));
    }

    #[test]
    fn test_request_serialize_with_body() {
        let body = b"Hello, World!";
        let request = Request::new(Method::Post, "/upload")
            .header("Host", "example.com")
            .body(BytesMut::from(&body[..]));

        let serialized = request.serialize();
        let serialized_str = String::from_utf8_lossy(&serialized);

        assert!(serialized_str.contains("POST /upload HTTP/1.1\r\n"));
        assert!(serialized_str.contains("Content-Length: 13\r\n"));
        assert!(serialized_str.ends_with("Hello, World!"));
    }
}

// =============================================================================
// HTTP Response Tests
// =============================================================================

#[cfg(test)]
mod http_response_tests {
    use super::*;

    #[test]
    fn test_response_new() {
        let response = Response::new(StatusCode::OK);

        assert_eq!(response.version, HttpVersion::Http11);
        assert_eq!(response.status, StatusCode::OK);
        assert!(response.body.is_empty());
    }

    #[test]
    fn test_response_builder_pattern() {
        let response = Response::new(StatusCode::CREATED)
            .header("Content-Type", "application/json")
            .header("Location", "/api/users/123")
            .body(BytesMut::from(&b"{\"id\":123}"[..]));

        assert_eq!(response.status, StatusCode::CREATED);
        assert_eq!(response.headers.get("Location"), Some("/api/users/123"));
        assert_eq!(&response.body[..], b"{\"id\":123}");
    }

    #[test]
    fn test_response_serialize() {
        let response = Response::new(StatusCode::OK)
            .header("Content-Type", "text/plain");

        let serialized = response.serialize();
        let serialized_str = String::from_utf8_lossy(&serialized);

        assert!(serialized_str.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(serialized_str.contains("content-type: text/plain\r\n"));
    }

    #[test]
    fn test_response_serialize_with_body() {
        let body = b"Response body";
        let response = Response::new(StatusCode::OK)
            .body(BytesMut::from(&body[..]));

        let serialized = response.serialize();
        let serialized_str = String::from_utf8_lossy(&serialized);

        assert!(serialized_str.contains("Content-Length: 13\r\n"));
        assert!(serialized_str.ends_with("Response body"));
    }

    #[test]
    fn test_response_different_status_codes() {
        let codes = vec![
            StatusCode::OK,
            StatusCode::CREATED,
            StatusCode::BAD_REQUEST,
            StatusCode::NOT_FOUND,
            StatusCode::INTERNAL_SERVER_ERROR,
        ];

        for code in codes {
            let response = Response::new(code);
            let serialized = response.serialize();
            let serialized_str = String::from_utf8_lossy(&serialized);

            assert!(serialized_str.contains(&format!("{} {}", code.0, code.reason_phrase())));
        }
    }
}

// =============================================================================
// HTTP Request Parser Tests
// =============================================================================

#[cfg(test)]
mod http_request_parser_tests {
    use super::*;

    #[test]
    fn test_parse_simple_get() {
        let mut parser = RequestParser::new();
        let data = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n";

        let request = parser.feed(data).unwrap().unwrap();

        assert_eq!(request.method, Method::Get);
        assert_eq!(request.uri, "/index.html");
        assert_eq!(request.version, HttpVersion::Http11);
        assert_eq!(request.headers.get("host"), Some("example.com"));
    }

    #[test]
    fn test_parse_post_with_body() {
        let mut parser = RequestParser::new();
        let data = b"POST /api HTTP/1.1\r\nHost: api.example.com\r\nContent-Length: 13\r\n\r\nHello, World!";

        let request = parser.feed(data).unwrap().unwrap();

        assert_eq!(request.method, Method::Post);
        assert_eq!(request.uri, "/api");
        assert_eq!(request.headers.get("content-length"), Some("13"));
        assert_eq!(&request.body[..], b"Hello, World!");
    }

    #[test]
    fn test_parse_multiple_headers() {
        let mut parser = RequestParser::new();
        let data = b"GET / HTTP/1.1\r\n\
                     Host: example.com\r\n\
                     User-Agent: TestClient\r\n\
                     Accept: text/html\r\n\
                     Accept-Language: en-US\r\n\
                     \r\n";

        let request = parser.feed(data).unwrap().unwrap();

        assert_eq!(request.headers.get("host"), Some("example.com"));
        assert_eq!(request.headers.get("user-agent"), Some("TestClient"));
        assert_eq!(request.headers.get("accept"), Some("text/html"));
        assert_eq!(request.headers.get("accept-language"), Some("en-US"));
    }

    #[test]
    fn test_parse_chunked_body() {
        let mut parser = RequestParser::new();
        let data = b"POST /upload HTTP/1.1\r\n\
                     Host: example.com\r\n\
                     Transfer-Encoding: chunked\r\n\
                     \r\n\
                     5\r\n\
                     Hello\r\n\
                     7\r\n\
                     , World\r\n\
                     0\r\n\
                     \r\n";

        let request = parser.feed(data).unwrap().unwrap();

        assert_eq!(request.method, Method::Post);
        assert_eq!(&request.body[..], b"Hello, World");
    }

    #[test]
    fn test_parse_chunked_with_extensions() {
        let mut parser = RequestParser::new();
        let data = b"POST /upload HTTP/1.1\r\n\
                     Transfer-Encoding: chunked\r\n\
                     \r\n\
                     5;ext=value\r\n\
                     Hello\r\n\
                     0\r\n\
                     \r\n";

        let request = parser.feed(data).unwrap().unwrap();
        assert_eq!(&request.body[..], b"Hello");
    }

    #[test]
    fn test_parse_incremental() {
        let mut parser = RequestParser::new();

        // Feed data in parts
        let result = parser.feed(b"GET /path ").unwrap();
        assert!(result.is_none()); // Not complete

        let result = parser.feed(b"HTTP/1.1\r\n").unwrap();
        assert!(result.is_none());

        let result = parser.feed(b"Host: example.com\r\n").unwrap();
        assert!(result.is_none());

        let result = parser.feed(b"\r\n").unwrap();
        assert!(result.is_some()); // Now complete

        let request = result.unwrap();
        assert_eq!(request.method, Method::Get);
        assert_eq!(request.uri, "/path");
    }

    #[test]
    fn test_parse_all_methods() {
        let methods = vec![
            ("GET", Method::Get),
            ("POST", Method::Post),
            ("PUT", Method::Put),
            ("DELETE", Method::Delete),
            ("HEAD", Method::Head),
            ("OPTIONS", Method::Options),
            ("PATCH", Method::Patch),
        ];

        for (method_str, expected) in methods {
            let mut parser = RequestParser::new();
            let data = format!("{} /path HTTP/1.1\r\nHost: test\r\n\r\n", method_str);

            let request = parser.feed(data.as_bytes()).unwrap().unwrap();
            assert_eq!(request.method, expected);
        }
    }

    #[test]
    fn test_parse_http10() {
        let mut parser = RequestParser::new();
        let data = b"GET /old HTTP/1.0\r\nHost: legacy.com\r\n\r\n";

        let request = parser.feed(data).unwrap().unwrap();
        assert_eq!(request.version, HttpVersion::Http10);
    }

    #[test]
    fn test_parse_empty_body() {
        let mut parser = RequestParser::new();
        let data = b"GET / HTTP/1.1\r\nHost: test\r\n\r\n";

        let request = parser.feed(data).unwrap().unwrap();
        assert!(request.body.is_empty());
    }

    #[test]
    fn test_parse_large_body() {
        let mut parser = RequestParser::new();
        let body = vec![b'X'; 10000];
        let mut data = format!("POST /upload HTTP/1.1\r\nContent-Length: {}\r\n\r\n", body.len())
            .into_bytes();
        data.extend_from_slice(&body);

        let request = parser.feed(&data).unwrap().unwrap();
        assert_eq!(request.body.len(), 10000);
    }

    #[test]
    fn test_parse_pipelined_requests() {
        let mut parser = RequestParser::new();
        let data = b"GET /first HTTP/1.1\r\nHost: test\r\n\r\n\
                     GET /second HTTP/1.1\r\nHost: test\r\n\r\n";

        // Parse first request
        let request1 = parser.feed(data).unwrap().unwrap();
        assert_eq!(request1.uri, "/first");

        // Parser should be ready for second request
        // (remaining data should be buffered)
    }

    #[test]
    fn test_parse_invalid_request_line() {
        let mut parser = RequestParser::new();
        let data = b"INVALID\r\n\r\n";

        let result = parser.feed(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_version() {
        let mut parser = RequestParser::new();
        let data = b"GET / HTTP/2.0\r\nHost: test\r\n\r\n";

        let result = parser.feed(data);
        assert!(result.is_err());
    }
}

// =============================================================================
// HTTP Response Parser Tests
// =============================================================================

#[cfg(test)]
mod http_response_parser_tests {
    use super::*;

    #[test]
    fn test_parse_simple_response() {
        let mut parser = ResponseParser::new();
        let data = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nHello";

        let response = parser.feed(data).unwrap().unwrap();

        assert_eq!(response.version, HttpVersion::Http11);
        assert_eq!(response.status.0, 200);
        assert_eq!(&response.body[..], b"Hello");
    }

    #[test]
    fn test_parse_response_all_status_codes() {
        let codes = vec![200, 201, 301, 400, 404, 500, 502];

        for code in codes {
            let mut parser = ResponseParser::new();
            let data = format!("HTTP/1.1 {} Reason\r\nContent-Length: 0\r\n\r\n", code);

            let response = parser.feed(data.as_bytes()).unwrap().unwrap();
            assert_eq!(response.status.0, code);
        }
    }

    #[test]
    fn test_parse_response_chunked() {
        let mut parser = ResponseParser::new();
        let data = b"HTTP/1.1 200 OK\r\n\
                     Transfer-Encoding: chunked\r\n\
                     \r\n\
                     7\r\n\
                     Mozilla\r\n\
                     9\r\n\
                     Developer\r\n\
                     7\r\n\
                     Network\r\n\
                     0\r\n\
                     \r\n";

        let response = parser.feed(data).unwrap().unwrap();
        assert_eq!(&response.body[..], b"MozillaDeveloperNetwork");
    }

    #[test]
    fn test_parse_response_no_content() {
        let mut parser = ResponseParser::new();
        let data = b"HTTP/1.1 204 No Content\r\n\r\n";

        let response = parser.feed(data).unwrap().unwrap();
        assert_eq!(response.status.0, 204);
        assert!(response.body.is_empty());
    }

    #[test]
    fn test_parse_response_with_headers() {
        let mut parser = ResponseParser::new();
        let data = b"HTTP/1.1 200 OK\r\n\
                     Content-Type: application/json\r\n\
                     Cache-Control: max-age=3600\r\n\
                     X-Custom-Header: custom-value\r\n\
                     Content-Length: 2\r\n\
                     \r\n\
                     {}";

        let response = parser.feed(data).unwrap().unwrap();

        assert_eq!(response.headers.get("content-type"), Some("application/json"));
        assert_eq!(response.headers.get("cache-control"), Some("max-age=3600"));
        assert_eq!(response.headers.get("x-custom-header"), Some("custom-value"));
    }

    #[test]
    fn test_parse_response_incremental() {
        let mut parser = ResponseParser::new();

        let result = parser.feed(b"HTTP/1.1 ").unwrap();
        assert!(result.is_none());

        let result = parser.feed(b"200 OK\r\n").unwrap();
        assert!(result.is_none());

        let result = parser.feed(b"Content-Length: 5\r\n").unwrap();
        assert!(result.is_none());

        let result = parser.feed(b"\r\n").unwrap();
        assert!(result.is_none()); // Still waiting for body

        let result = parser.feed(b"Hello").unwrap();
        assert!(result.is_some());
    }
}

// =============================================================================
// URL Parser Tests
// =============================================================================

#[cfg(test)]
mod url_parser_tests {
    use super::*;

    #[test]
    fn test_parse_simple_url() {
        let url = Url::parse("http://example.com/path").unwrap();

        assert_eq!(url.scheme, "http");
        assert_eq!(url.host, "example.com");
        assert_eq!(url.port, None);
        assert_eq!(url.path, "/path");
        assert_eq!(url.query, None);
    }

    #[test]
    fn test_parse_url_with_port() {
        let url = Url::parse("https://example.com:8080/path").unwrap();

        assert_eq!(url.scheme, "https");
        assert_eq!(url.host, "example.com");
        assert_eq!(url.port, Some(8080));
        assert_eq!(url.path, "/path");
    }

    #[test]
    fn test_parse_url_with_query() {
        let url = Url::parse("http://example.com/search?q=test&page=1").unwrap();

        assert_eq!(url.path, "/search");
        assert_eq!(url.query, Some("q=test&page=1".to_string()));
    }

    #[test]
    fn test_parse_url_root_path() {
        let url = Url::parse("http://example.com").unwrap();

        assert_eq!(url.path, "/");
        assert_eq!(url.query, None);
    }

    #[test]
    fn test_parse_url_no_scheme() {
        let result = Url::parse("example.com/path");
        assert!(result.is_err());
    }

    #[test]
    fn test_effective_port_http() {
        let url = Url::parse("http://example.com/").unwrap();
        assert_eq!(url.effective_port(), 80);
    }

    #[test]
    fn test_effective_port_https() {
        let url = Url::parse("https://example.com/").unwrap();
        assert_eq!(url.effective_port(), 443);
    }

    #[test]
    fn test_effective_port_explicit() {
        let url = Url::parse("http://example.com:3000/").unwrap();
        assert_eq!(url.effective_port(), 3000);
    }

    #[test]
    fn test_host_port() {
        let url = Url::parse("http://example.com/").unwrap();
        assert_eq!(url.host_port(), "example.com");

        let url_with_port = Url::parse("http://example.com:8080/").unwrap();
        assert_eq!(url_with_port.host_port(), "example.com:8080");
    }

    #[test]
    fn test_path_and_query() {
        let url = Url::parse("http://example.com/path").unwrap();
        assert_eq!(url.path_and_query(), "/path");

        let url_with_query = Url::parse("http://example.com/search?q=test").unwrap();
        assert_eq!(url_with_query.path_and_query(), "/search?q=test");
    }

    #[test]
    fn test_parse_complex_url() {
        let url = Url::parse("https://api.example.com:443/v1/users?limit=10&offset=20").unwrap();

        assert_eq!(url.scheme, "https");
        assert_eq!(url.host, "api.example.com");
        assert_eq!(url.port, Some(443));
        assert_eq!(url.path, "/v1/users");
        assert_eq!(url.query, Some("limit=10&offset=20".to_string()));
    }

    #[test]
    fn test_parse_url_invalid_port() {
        let result = Url::parse("http://example.com:invalid/");
        assert!(result.is_err());
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

#[cfg(test)]
mod http_integration_tests {
    use super::*;

    #[test]
    fn test_roundtrip_request() {
        // Create request
        let original = Request::new(Method::Post, "/api/data")
            .header("Host", "api.example.com")
            .header("Content-Type", "application/json")
            .body(BytesMut::from(&b"{\"key\":\"value\"}"[..]));

        // Serialize
        let serialized = original.serialize();

        // Parse back
        let mut parser = RequestParser::new();
        let parsed = parser.feed(&serialized).unwrap().unwrap();

        // Verify
        assert_eq!(parsed.method, original.method);
        assert_eq!(parsed.uri, original.uri);
        assert_eq!(parsed.version, original.version);
        assert_eq!(&parsed.body[..], &original.body[..]);
    }

    #[test]
    fn test_roundtrip_response() {
        // Create response
        let original = Response::new(StatusCode::OK)
            .header("Content-Type", "text/plain")
            .header("X-Custom", "value")
            .body(BytesMut::from(&b"Response body"[..]));

        // Serialize
        let serialized = original.serialize();

        // Parse back
        let mut parser = ResponseParser::new();
        let parsed = parser.feed(&serialized).unwrap().unwrap();

        // Verify
        assert_eq!(parsed.status, original.status);
        assert_eq!(parsed.version, original.version);
        assert_eq!(&parsed.body[..], &original.body[..]);
    }

    #[test]
    fn test_request_response_cycle() {
        // Simulate HTTP request/response cycle

        // Build request
        let request = Request::new(Method::Get, "/api/status")
            .header("Host", "api.example.com")
            .header("Accept", "application/json");

        // Serialize request
        let request_bytes = request.serialize();

        // Server parses request
        let mut request_parser = RequestParser::new();
        let parsed_request = request_parser.feed(&request_bytes).unwrap().unwrap();

        assert_eq!(parsed_request.method, Method::Get);
        assert_eq!(parsed_request.uri, "/api/status");

        // Server builds response
        let response = Response::new(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(BytesMut::from(&b"{\"status\":\"ok\"}"[..]));

        // Serialize response
        let response_bytes = response.serialize();

        // Client parses response
        let mut response_parser = ResponseParser::new();
        let parsed_response = response_parser.feed(&response_bytes).unwrap().unwrap();

        assert_eq!(parsed_response.status, StatusCode::OK);
        assert_eq!(&parsed_response.body[..], b"{\"status\":\"ok\"}");
    }
}
