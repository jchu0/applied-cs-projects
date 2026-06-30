fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile protobuf definitions
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile(&["proto/kafka.proto"], &["proto/"])?;
    Ok(())
}
