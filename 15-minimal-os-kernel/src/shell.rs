//! Simple shell for the minimal OS kernel.
//!
//! Provides a basic command-line interface with:
//! - Command parsing and execution
//! - Built-in commands (cd, pwd, exit, echo)
//! - External command execution via fork/exec
//! - Basic I/O redirection
//! - Simple job control

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::vec;

use crate::process::Pid;
use crate::scheduler;
use crate::vfs;
use crate::serial_println;
use crate::serial_print;

/// Maximum command line length.
const MAX_CMD_LEN: usize = 256;

/// Shell prompt.
const PROMPT: &str = "$ ";

/// Shell main loop.
pub fn shell_main() {
    serial_println!("MinOS Shell v0.1");
    serial_println!("Type 'help' for available commands.");
    serial_println!("");

    let mut cwd = String::from("/");

    loop {
        // Print prompt
        serial_print!("{}{}", cwd, PROMPT);

        // Read command
        let input = read_line();

        if input.is_empty() {
            continue;
        }

        // Parse and execute
        let result = execute_command(&input, &mut cwd);

        // Handle exit
        if let CommandResult::Exit(code) = result {
            serial_println!("exit {}", code);
            scheduler::exit(code);
            return;
        }
    }
}

/// Result of command execution.
#[derive(Debug)]
enum CommandResult {
    Success,
    Error(String),
    Exit(i32),
}

/// Read a line from input.
fn read_line() -> String {
    let mut buffer = String::new();

    loop {
        // In a real implementation, this would read from keyboard/console
        // For now, we use serial input
        if let Some(c) = read_char() {
            match c {
                '\n' | '\r' => {
                    serial_println!("");
                    break;
                }
                '\x08' | '\x7f' => {
                    // Backspace
                    if !buffer.is_empty() {
                        buffer.pop();
                        serial_print!("\x08 \x08"); // Erase character
                    }
                }
                c if c.is_ascii() && buffer.len() < MAX_CMD_LEN => {
                    buffer.push(c);
                    serial_print!("{}", c);
                }
                _ => {}
            }
        }

        // Yield to prevent busy waiting
        scheduler::yield_cpu();
    }

    buffer
}

/// Read a single character from input.
fn read_char() -> Option<char> {
    // This would be connected to keyboard/console driver
    // For now, return None (no input)
    None
}

/// Execute a command.
fn execute_command(input: &str, cwd: &mut String) -> CommandResult {
    let args = parse_command(input);

    if args.is_empty() {
        return CommandResult::Success;
    }

    let cmd = &args[0];
    let cmd_args = &args[1..];

    // Handle built-in commands
    match cmd.as_str() {
        "exit" | "quit" => {
            let code = cmd_args.get(0)
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(0);
            CommandResult::Exit(code)
        }
        "cd" => builtin_cd(cmd_args, cwd),
        "pwd" => builtin_pwd(cwd),
        "echo" => builtin_echo(cmd_args),
        "cat" => builtin_cat(cmd_args),
        "ls" => builtin_ls(cmd_args, cwd),
        "mkdir" => builtin_mkdir(cmd_args),
        "touch" => builtin_touch(cmd_args),
        "rm" => builtin_rm(cmd_args),
        "help" => builtin_help(),
        "clear" => builtin_clear(),
        "ps" => builtin_ps(),
        "kill" => builtin_kill(cmd_args),
        _ => execute_external(cmd, cmd_args),
    }
}

/// Parse command into arguments.
fn parse_command(input: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut escape_next = false;

    for c in input.chars() {
        if escape_next {
            current.push(c);
            escape_next = false;
            continue;
        }

        match c {
            '\\' => escape_next = true,
            '"' => in_quotes = !in_quotes,
            ' ' | '\t' if !in_quotes => {
                if !current.is_empty() {
                    args.push(current.clone());
                    current.clear();
                }
            }
            _ => current.push(c),
        }
    }

    if !current.is_empty() {
        args.push(current);
    }

    args
}

/// Built-in: cd - change directory.
fn builtin_cd(args: &[String], cwd: &mut String) -> CommandResult {
    let path = args.get(0).map(|s| s.as_str()).unwrap_or("/");

    let new_path = if path.starts_with('/') {
        path.to_string()
    } else {
        format!("{}/{}", cwd, path)
    };

    // Normalize path
    let normalized = normalize_path(&new_path);

    // Check if directory exists
    match vfs::stat(&normalized) {
        Ok(stat) if vfs::is_directory(&stat) => {
            *cwd = normalized;
            CommandResult::Success
        }
        Ok(_) => CommandResult::Error(format!("{}: Not a directory", path)),
        Err(_) => CommandResult::Error(format!("{}: No such directory", path)),
    }
}

/// Built-in: pwd - print working directory.
fn builtin_pwd(cwd: &String) -> CommandResult {
    serial_println!("{}", cwd);
    CommandResult::Success
}

/// Built-in: echo - print arguments.
fn builtin_echo(args: &[String]) -> CommandResult {
    let output = args.join(" ");
    serial_println!("{}", output);
    CommandResult::Success
}

/// Built-in: cat - concatenate and display files.
fn builtin_cat(args: &[String]) -> CommandResult {
    if args.is_empty() {
        return CommandResult::Error("Usage: cat <file>...".to_string());
    }

    for path in args {
        match vfs::open(path, vfs::O_RDONLY, 0) {
            Ok(fd) => {
                let mut buffer = [0u8; 512];
                loop {
                    match vfs::read_fd(fd, &mut buffer) {
                        Ok(0) => break,
                        Ok(n) => {
                            if let Ok(s) = core::str::from_utf8(&buffer[..n]) {
                                serial_print!("{}", s);
                            }
                        }
                        Err(e) => {
                            serial_println!("cat: {}: {:?}", path, e);
                            break;
                        }
                    }
                }
                let _ = vfs::close(fd);
            }
            Err(e) => {
                serial_println!("cat: {}: {:?}", path, e);
            }
        }
    }

    CommandResult::Success
}

/// Built-in: ls - list directory contents.
fn builtin_ls(args: &[String], cwd: &String) -> CommandResult {
    let path = args.get(0).map(|s| s.as_str()).unwrap_or(cwd.as_str());

    match vfs::readdir(path) {
        Ok(entries) => {
            for entry in entries {
                let type_char = if entry.is_dir { 'd' } else { '-' };
                serial_println!("{}  {}", type_char, entry.name);
            }
            CommandResult::Success
        }
        Err(e) => CommandResult::Error(format!("ls: {}: {:?}", path, e)),
    }
}

/// Built-in: mkdir - make directories.
fn builtin_mkdir(args: &[String]) -> CommandResult {
    if args.is_empty() {
        return CommandResult::Error("Usage: mkdir <dir>...".to_string());
    }

    for path in args {
        if let Err(e) = vfs::mkdir(path, 0o755) {
            serial_println!("mkdir: {}: {:?}", path, e);
        }
    }

    CommandResult::Success
}

/// Built-in: touch - create empty files.
fn builtin_touch(args: &[String]) -> CommandResult {
    if args.is_empty() {
        return CommandResult::Error("Usage: touch <file>...".to_string());
    }

    for path in args {
        match vfs::open(path, vfs::O_CREAT | vfs::O_WRONLY, 0o644) {
            Ok(fd) => { let _ = vfs::close(fd); }
            Err(e) => serial_println!("touch: {}: {:?}", path, e),
        }
    }

    CommandResult::Success
}

/// Built-in: rm - remove files.
fn builtin_rm(args: &[String]) -> CommandResult {
    if args.is_empty() {
        return CommandResult::Error("Usage: rm <file>...".to_string());
    }

    for path in args {
        if let Err(e) = vfs::unlink(path) {
            serial_println!("rm: {}: {:?}", path, e);
        }
    }

    CommandResult::Success
}

/// Built-in: help - display help.
fn builtin_help() -> CommandResult {
    serial_println!("MinOS Shell - Available Commands:");
    serial_println!("");
    serial_println!("  cd <dir>      Change directory");
    serial_println!("  pwd           Print working directory");
    serial_println!("  ls [dir]      List directory contents");
    serial_println!("  cat <file>    Display file contents");
    serial_println!("  echo <text>   Print text");
    serial_println!("  mkdir <dir>   Create directory");
    serial_println!("  touch <file>  Create empty file");
    serial_println!("  rm <file>     Remove file");
    serial_println!("  ps            List processes");
    serial_println!("  kill <pid>    Kill process");
    serial_println!("  clear         Clear screen");
    serial_println!("  exit [code]   Exit shell");
    serial_println!("  help          Show this help");
    serial_println!("");

    CommandResult::Success
}

/// Built-in: clear - clear screen.
fn builtin_clear() -> CommandResult {
    // ANSI escape sequence to clear screen
    serial_print!("\x1b[2J\x1b[H");
    CommandResult::Success
}

/// Built-in: ps - list processes.
fn builtin_ps() -> CommandResult {
    serial_println!("  PID  STATE    NAME");

    for process in scheduler::list_processes() {
        let state = match process.state {
            crate::process::ProcessState::Ready => "READY",
            crate::process::ProcessState::Running => "RUN  ",
            crate::process::ProcessState::Blocked(_) => "BLOCK",
            crate::process::ProcessState::Zombie => "ZOMB ",
            crate::process::ProcessState::Stopped => "STOP ",
        };
        serial_println!("{:5}  {}  {}", process.pid, state, process.name);
    }

    CommandResult::Success
}

/// Built-in: kill - send signal to process.
fn builtin_kill(args: &[String]) -> CommandResult {
    if args.is_empty() {
        return CommandResult::Error("Usage: kill <pid>".to_string());
    }

    let pid: Pid = match args[0].parse() {
        Ok(p) => p,
        Err(_) => return CommandResult::Error("Invalid PID".to_string()),
    };

    // Default signal is SIGTERM (15)
    let signal = 15;

    match scheduler::kill(pid, signal) {
        Ok(()) => CommandResult::Success,
        Err(e) => CommandResult::Error(format!("kill {}: {:?}", pid, e)),
    }
}

/// Execute external command.
fn execute_external(cmd: &str, args: &[String]) -> CommandResult {
    // Search for command in /bin
    let path = format!("/bin/{}", cmd);

    // Fork
    let pid = match scheduler::fork() {
        Ok(p) => p,
        Err(e) => return CommandResult::Error(format!("fork: {:?}", e)),
    };

    if pid == 0 {
        // Child process - exec
        let argv: Vec<&str> = core::iter::once(cmd)
            .chain(args.iter().map(|s| s.as_str()))
            .collect();

        match scheduler::execve(&path, &argv, &[]) {
            Ok(()) => {
                // execve doesn't return on success
                unreachable!()
            }
            Err(e) => {
                serial_println!("{}: {:?}", cmd, e);
                scheduler::exit(127);
            }
        }
    }

    // Parent - wait for child
    match scheduler::waitpid(pid, 0) {
        Ok((_, status)) => {
            if status != 0 {
                CommandResult::Error(format!("Exit status: {}", status))
            } else {
                CommandResult::Success
            }
        }
        Err(e) => CommandResult::Error(format!("wait: {:?}", e)),
    }
}

/// Normalize a path (resolve . and ..).
fn normalize_path(path: &str) -> String {
    let mut components: Vec<&str> = Vec::new();

    for component in path.split('/') {
        match component {
            "" | "." => continue,
            ".." => { components.pop(); }
            c => components.push(c),
        }
    }

    if components.is_empty() {
        "/".to_string()
    } else {
        format!("/{}", components.join("/"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_command() {
        let args = parse_command("ls -la");
        assert_eq!(args, vec!["ls", "-la"]);
    }

    #[test]
    fn test_parse_quoted_args() {
        let args = parse_command("echo \"hello world\"");
        assert_eq!(args, vec!["echo", "hello world"]);
    }

    #[test]
    fn test_parse_empty() {
        let args = parse_command("");
        assert!(args.is_empty());
    }

    #[test]
    fn test_parse_whitespace() {
        let args = parse_command("   ");
        assert!(args.is_empty());
    }

    #[test]
    fn test_normalize_path_simple() {
        assert_eq!(normalize_path("/home/user"), "/home/user");
    }

    #[test]
    fn test_normalize_path_dots() {
        assert_eq!(normalize_path("/home/user/../admin"), "/home/admin");
    }

    #[test]
    fn test_normalize_path_current() {
        assert_eq!(normalize_path("/home/./user"), "/home/user");
    }

    #[test]
    fn test_normalize_path_root() {
        assert_eq!(normalize_path("/.."), "/");
    }
}
