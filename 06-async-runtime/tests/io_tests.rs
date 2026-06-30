//! Test suite for async I/O operations

use std::io::{self, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use async_runtime::io::{AsyncRead, AsyncWrite, AsyncBufRead, AsyncSeek};
use async_runtime::net::{TcpListener, TcpStream, UdpSocket};
use async_runtime::fs::{File, OpenOptions};
use async_runtime::runtime::Runtime;

#[test]
fn test_tcp_echo_server() {
    let runtime = Runtime::new().unwrap();

    runtime.block_on(async {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let listener = TcpListener::bind(addr).await.unwrap();
        let server_addr = listener.local_addr().unwrap();

        // Spawn echo server
        async_runtime::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 1024];

            loop {
                match stream.read(&mut buf).await {
                    Ok(0) => break, // Connection closed
                    Ok(n) => {
                        stream.write_all(&buf[..n]).await.unwrap();
                    }
                    Err(_) => break,
                }
            }
        });

        // Client
        let mut client = TcpStream::connect(server_addr).await.unwrap();

        let message = b"Hello, async world!";
        client.write_all(message).await.unwrap();

        let mut response = vec![0u8; message.len()];
        client.read_exact(&mut response).await.unwrap();

        assert_eq!(&response, message);
    });
}

#[test]
fn test_udp_socket() {
    let runtime = Runtime::new().unwrap();

    runtime.block_on(async {
        let server_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let server = UdpSocket::bind(server_addr).await.unwrap();
        let server_addr = server.local_addr().unwrap();

        let client_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let client = UdpSocket::bind(client_addr).await.unwrap();

        // Server task
        async_runtime::spawn(async move {
            let mut buf = [0u8; 1024];
            let (n, from) = server.recv_from(&mut buf).await.unwrap();
            server.send_to(&buf[..n], from).await.unwrap();
        });

        // Send message
        let message = b"UDP test message";
        client.send_to(message, server_addr).await.unwrap();

        // Receive echo
        let mut buf = [0u8; 1024];
        let (n, _) = client.recv_from(&mut buf).await.unwrap();

        assert_eq!(&buf[..n], message);
    });
}

#[test]
fn test_async_file_io() {
    let runtime = Runtime::new().unwrap();

    runtime.block_on(async {
        let temp_file = std::env::temp_dir().join("async_test_file.txt");

        // Write to file
        {
            let mut file = File::create(&temp_file).await.unwrap();
            file.write_all(b"Hello, async file I/O!\n").await.unwrap();
            file.write_all(b"Second line\n").await.unwrap();
            file.sync_all().await.unwrap();
        }

        // Read from file
        {
            let mut file = File::open(&temp_file).await.unwrap();
            let mut contents = Vec::new();
            file.read_to_end(&mut contents).await.unwrap();

            assert_eq!(contents, b"Hello, async file I/O!\nSecond line\n");
        }

        // Seek and read
        {
            let mut file = File::open(&temp_file).await.unwrap();
            file.seek(io::SeekFrom::Start(7)).await.unwrap();

            let mut buf = [0u8; 5];
            file.read_exact(&mut buf).await.unwrap();

            assert_eq!(&buf, b"async");
        }

        // Clean up
        async_runtime::fs::remove_file(temp_file).await.unwrap();
    });
}

#[test]
fn test_buffered_io() {
    let runtime = Runtime::new().unwrap();

    runtime.block_on(async {
        let temp_file = std::env::temp_dir().join("buffered_test.txt");

        // Write lines
        {
            let file = File::create(&temp_file).await.unwrap();
            let mut writer = async_runtime::io::BufWriter::new(file);

            for i in 0..100 {
                writeln!(writer, "Line {}", i).await.unwrap();
            }

            writer.flush().await.unwrap();
        }

        // Read lines
        {
            let file = File::open(&temp_file).await.unwrap();
            let reader = async_runtime::io::BufReader::new(file);

            let mut lines = reader.lines();
            let mut count = 0;

            while let Some(line) = lines.next().await {
                let line = line.unwrap();
                assert_eq!(line, format!("Line {}", count));
                count += 1;
            }

            assert_eq!(count, 100);
        }

        async_runtime::fs::remove_file(temp_file).await.unwrap();
    });
}

#[test]
fn test_concurrent_io() {
    let runtime = Runtime::new().unwrap();

    runtime.block_on(async {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let listener = TcpListener::bind(addr).await.unwrap();
        let server_addr = listener.local_addr().unwrap();

        // Spawn server that handles multiple connections
        async_runtime::spawn(async move {
            for _ in 0..5 {
                let (mut stream, _) = listener.accept().await.unwrap();

                async_runtime::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let n = stream.read(&mut buf).await.unwrap();
                    stream.write_all(&buf[..n]).await.unwrap();
                });
            }
        });

        // Connect multiple clients concurrently
        let mut handles = Vec::new();

        for i in 0..5 {
            let handle = async_runtime::spawn(async move {
                let mut client = TcpStream::connect(server_addr).await.unwrap();

                let message = format!("Client {}", i);
                client.write_all(message.as_bytes()).await.unwrap();

                let mut response = vec![0u8; message.len()];
                client.read_exact(&mut response).await.unwrap();

                String::from_utf8(response).unwrap()
            });

            handles.push(handle);
        }

        for (i, handle) in handles.into_iter().enumerate() {
            let response = handle.await;
            assert_eq!(response, format!("Client {}", i));
        }
    });
}

#[test]
fn test_io_timeout() {
    let runtime = Runtime::new().unwrap();

    runtime.block_on(async {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let listener = TcpListener::bind(addr).await.unwrap();
        let server_addr = listener.local_addr().unwrap();

        // Server that never responds
        async_runtime::spawn(async move {
            let (_stream, _) = listener.accept().await.unwrap();
            // Just hold the connection open
            async_runtime::time::sleep(Duration::from_secs(10)).await;
        });

        let mut client = TcpStream::connect(server_addr).await.unwrap();
        client.set_read_timeout(Some(Duration::from_millis(100))).unwrap();

        let mut buf = [0u8; 1024];
        let result = async_runtime::time::timeout(
            Duration::from_millis(200),
            client.read(&mut buf)
        ).await;

        assert!(result.is_err() || result.unwrap().is_err());
    });
}

#[test]
fn test_async_stdio() {
    let runtime = Runtime::new().unwrap();

    runtime.block_on(async {
        use async_runtime::io::{stdin, stdout, stderr};

        // Test that we can create async stdio handles
        let _stdin = stdin();
        let _stdout = stdout();
        let _stderr = stderr();

        // Note: Actually testing stdio would require process spawning
        // and pipe redirection, which is beyond this test's scope
    });
}

#[test]
fn test_file_metadata() {
    let runtime = Runtime::new().unwrap();

    runtime.block_on(async {
        let temp_file = std::env::temp_dir().join("metadata_test.txt");

        let mut file = File::create(&temp_file).await.unwrap();
        file.write_all(b"Test content").await.unwrap();
        file.sync_all().await.unwrap();

        let metadata = file.metadata().await.unwrap();
        assert!(metadata.is_file());
        assert!(!metadata.is_dir());
        assert_eq!(metadata.len(), 12); // "Test content" is 12 bytes

        // Test permissions
        let mut perms = metadata.permissions();
        perms.set_readonly(true);
        file.set_permissions(perms).await.unwrap();

        let metadata = file.metadata().await.unwrap();
        assert!(metadata.permissions().readonly());

        // Clean up
        drop(file);
        let mut perms = metadata.permissions();
        perms.set_readonly(false);
        async_runtime::fs::set_permissions(&temp_file, perms).await.unwrap();
        async_runtime::fs::remove_file(temp_file).await.unwrap();
    });
}

#[test]
fn test_directory_operations() {
    let runtime = Runtime::new().unwrap();

    runtime.block_on(async {
        let temp_dir = std::env::temp_dir().join("async_dir_test");

        // Create directory
        async_runtime::fs::create_dir(&temp_dir).await.unwrap();

        // Create files in directory
        for i in 0..5 {
            let file_path = temp_dir.join(format!("file_{}.txt", i));
            let mut file = File::create(&file_path).await.unwrap();
            file.write_all(format!("Content {}", i).as_bytes()).await.unwrap();
        }

        // Read directory
        let mut entries = async_runtime::fs::read_dir(&temp_dir).await.unwrap();
        let mut count = 0;

        while let Some(entry) = entries.next().await {
            let entry = entry.unwrap();
            assert!(entry.file_name().to_str().unwrap().starts_with("file_"));
            count += 1;
        }

        assert_eq!(count, 5);

        // Remove directory and contents
        async_runtime::fs::remove_dir_all(temp_dir).await.unwrap();
    });
}

#[test]
fn test_copy_file() {
    let runtime = Runtime::new().unwrap();

    runtime.block_on(async {
        let source = std::env::temp_dir().join("source_file.txt");
        let destination = std::env::temp_dir().join("destination_file.txt");

        // Create source file
        let mut file = File::create(&source).await.unwrap();
        file.write_all(b"Content to copy").await.unwrap();
        drop(file);

        // Copy file
        let bytes_copied = async_runtime::fs::copy(&source, &destination).await.unwrap();
        assert_eq!(bytes_copied, 15); // "Content to copy" is 15 bytes

        // Verify copy
        let mut file = File::open(&destination).await.unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).await.unwrap();
        assert_eq!(contents, "Content to copy");

        // Clean up
        async_runtime::fs::remove_file(source).await.unwrap();
        async_runtime::fs::remove_file(destination).await.unwrap();
    });
}

#[test]
fn test_split_io() {
    let runtime = Runtime::new().unwrap();

    runtime.block_on(async {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let listener = TcpListener::bind(addr).await.unwrap();
        let server_addr = listener.local_addr().unwrap();

        // Echo server
        async_runtime::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (mut reader, mut writer) = stream.split();

            async_runtime::io::copy(&mut reader, &mut writer).await.unwrap();
        });

        // Client with split streams
        let stream = TcpStream::connect(server_addr).await.unwrap();
        let (mut reader, mut writer) = stream.split();

        // Write in one task
        let write_handle = async_runtime::spawn(async move {
            writer.write_all(b"Split I/O test").await.unwrap();
            writer.shutdown().await.unwrap();
        });

        // Read in another task
        let read_handle = async_runtime::spawn(async move {
            let mut buf = Vec::new();
            reader.read_to_end(&mut buf).await.unwrap();
            buf
        });

        write_handle.await;
        let received = read_handle.await;

        assert_eq!(received, b"Split I/O test");
    });
}