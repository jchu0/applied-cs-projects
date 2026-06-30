//! Global Descriptor Table (GDT) and Task State Segment (TSS).

use lazy_static::lazy_static;
use x86_64::structures::gdt::{Descriptor, GlobalDescriptorTable, SegmentSelector};
use x86_64::structures::tss::TaskStateSegment;
use x86_64::VirtAddr;

/// Double fault stack index in IST.
pub const DOUBLE_FAULT_IST_INDEX: u16 = 0;

/// Stack size for interrupt stacks.
const STACK_SIZE: usize = 4096 * 5;

lazy_static! {
    /// Task State Segment.
    static ref TSS: TaskStateSegment = {
        let mut tss = TaskStateSegment::new();

        // Set up interrupt stack for double faults
        tss.interrupt_stack_table[DOUBLE_FAULT_IST_INDEX as usize] = {
            static mut STACK: [u8; STACK_SIZE] = [0; STACK_SIZE];
            let stack_start = VirtAddr::from_ptr(unsafe { &STACK });
            let stack_end = stack_start + STACK_SIZE;
            stack_end
        };

        // Set up privilege stack for ring 0
        tss.privilege_stack_table[0] = {
            static mut STACK: [u8; STACK_SIZE] = [0; STACK_SIZE];
            let stack_start = VirtAddr::from_ptr(unsafe { &STACK });
            let stack_end = stack_start + STACK_SIZE;
            stack_end
        };

        tss
    };
}

lazy_static! {
    /// Global Descriptor Table.
    static ref GDT: (GlobalDescriptorTable, Selectors) = {
        let mut gdt = GlobalDescriptorTable::new();

        let code_selector = gdt.add_entry(Descriptor::kernel_code_segment());
        let data_selector = gdt.add_entry(Descriptor::kernel_data_segment());
        let tss_selector = gdt.add_entry(Descriptor::tss_segment(&TSS));
        let user_data_selector = gdt.add_entry(Descriptor::user_data_segment());
        let user_code_selector = gdt.add_entry(Descriptor::user_code_segment());

        (
            gdt,
            Selectors {
                code_selector,
                data_selector,
                tss_selector,
                user_code_selector,
                user_data_selector,
            },
        )
    };
}

/// Segment selectors.
struct Selectors {
    code_selector: SegmentSelector,
    data_selector: SegmentSelector,
    tss_selector: SegmentSelector,
    user_code_selector: SegmentSelector,
    user_data_selector: SegmentSelector,
}

/// Initialize GDT.
pub fn init() {
    use x86_64::instructions::segmentation::{Segment, CS, DS, ES, FS, GS, SS};
    use x86_64::instructions::tables::load_tss;

    GDT.0.load();

    unsafe {
        CS::set_reg(GDT.1.code_selector);
        DS::set_reg(GDT.1.data_selector);
        ES::set_reg(GDT.1.data_selector);
        FS::set_reg(GDT.1.data_selector);
        GS::set_reg(GDT.1.data_selector);
        SS::set_reg(GDT.1.data_selector);
        load_tss(GDT.1.tss_selector);
    }
}

/// Get kernel code selector.
pub fn get_kernel_code_selector() -> SegmentSelector {
    GDT.1.code_selector
}

/// Get kernel data selector.
pub fn get_kernel_data_selector() -> SegmentSelector {
    GDT.1.data_selector
}

/// Get user code selector.
pub fn get_user_code_selector() -> SegmentSelector {
    GDT.1.user_code_selector
}

/// Get user data selector.
pub fn get_user_data_selector() -> SegmentSelector {
    GDT.1.user_data_selector
}
