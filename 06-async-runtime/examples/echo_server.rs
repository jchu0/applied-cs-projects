//! Echo server example
//!
//! A simple TCP echo server that demonstrates async I/O.

use async_runtime::net::TcpListener;
use async_runtime::{spawn, Runtime};
use std::net::SocketAddr;

fn main() -> std::io::Result<()> {
    let rt = Runtime::new()?;

    rt.block_on(async {
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let listener = TcpListener::bind(addr)?;
        println!("Echo server listening on {}", addr);

        loop {
            let (mut stream, peer_addr) = listener.accept().await?;
            println!("Connection from {}", peer_addr);

            spawn(async move {
                let mut buf = [0u8; 1024];
                loop {
                    let n = match stream.read(&mut buf).await {
                        Ok(0) => {
                            println!("Connection closed by {}", peer_addr);
                            return;
                        }
                        Ok(n) => n,
                        Err(e) => {
                            eprintln!("Read error from {}: {}", peer_addr, e);
                            return;
                        }
                    };

                    if let Err(e) = stream.write_all(&buf[..n]).await {
                        eprintln!("Write error to {}: {}", peer_addr, e);
                        return;
                    }
                }
            });
        }
    })
}
