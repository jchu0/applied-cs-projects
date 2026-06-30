use py_compiler::run;

#[test]
fn debug_algorithm_sum_list() {
    let code = r#"
def sum_list(items):
    total = 0
    for item in items:
        total = total + item
    return total
result = sum_list([1, 2, 3, 4, 5])
"#;
    match run(code) {
        Ok(v) => println!("Success: {:?}", v),
        Err(e) => panic!("Error: {}", e),
    }
}
