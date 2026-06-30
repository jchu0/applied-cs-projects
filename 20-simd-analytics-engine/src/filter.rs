//! Vectorized filtering operations.

use crate::column::{AlignedVec, Column};
use crate::{Error, Result, VECTOR_WIDTH};

/// Filter comparison operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterOp {
    /// Equal
    Eq,
    /// Not equal
    Ne,
    /// Greater than
    Gt,
    /// Greater than or equal
    Ge,
    /// Less than
    Lt,
    /// Less than or equal
    Le,
}

/// Filter predicate with comparison value.
#[derive(Debug, Clone)]
pub enum FilterPredicate {
    Int32(FilterOp, i32),
    Int64(FilterOp, i64),
    Float32(FilterOp, f32),
    Float64(FilterOp, f64),
}

/// Selection bitmap for filtered rows.
#[derive(Debug, Clone)]
pub struct SelectionVector {
    /// Bitmap where each bit indicates if row passes filter.
    bitmap: Vec<u64>,
    /// Number of rows.
    num_rows: usize,
    /// Number of selected rows.
    selected_count: usize,
}

impl SelectionVector {
    /// Create new selection vector.
    pub fn new(num_rows: usize) -> Self {
        let num_words = (num_rows + 63) / 64;
        Self {
            bitmap: vec![0; num_words],
            num_rows,
            selected_count: 0,
        }
    }

    /// Create with all rows selected.
    pub fn all_selected(num_rows: usize) -> Self {
        let num_words = (num_rows + 63) / 64;
        let mut bitmap = vec![!0u64; num_words];

        // Clear extra bits in last word
        let extra_bits = num_rows % 64;
        if extra_bits > 0 {
            bitmap[num_words - 1] = (1u64 << extra_bits) - 1;
        }

        Self {
            bitmap,
            num_rows,
            selected_count: num_rows,
        }
    }

    /// Check if row is selected.
    #[inline]
    pub fn is_selected(&self, index: usize) -> bool {
        if index >= self.num_rows {
            return false;
        }
        let word = index / 64;
        let bit = index % 64;
        (self.bitmap[word] >> bit) & 1 == 1
    }

    /// Set row selection.
    #[inline]
    pub fn set(&mut self, index: usize, selected: bool) {
        if index >= self.num_rows {
            return;
        }
        let word = index / 64;
        let bit = index % 64;
        if selected {
            self.bitmap[word] |= 1u64 << bit;
        } else {
            self.bitmap[word] &= !(1u64 << bit);
        }
    }

    /// Get number of selected rows.
    pub fn count(&self) -> usize {
        self.selected_count
    }

    /// Get total number of rows.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Recount selected rows.
    pub fn recount(&mut self) {
        self.selected_count = self.bitmap.iter().map(|w| w.count_ones() as usize).sum();
    }

    /// Get bitmap word at index.
    pub fn get_word(&self, word_index: usize) -> u64 {
        self.bitmap.get(word_index).copied().unwrap_or(0)
    }

    /// Set bitmap word at index.
    pub fn set_word(&mut self, word_index: usize, value: u64) {
        if word_index < self.bitmap.len() {
            self.bitmap[word_index] = value;
        }
    }

    /// Number of bitmap words.
    pub fn num_words(&self) -> usize {
        self.bitmap.len()
    }

    /// AND two selection vectors.
    pub fn and(&self, other: &SelectionVector) -> Result<SelectionVector> {
        if self.num_rows != other.num_rows {
            return Err(Error::DimensionMismatch(format!(
                "Selection vector sizes don't match: {} vs {}",
                self.num_rows, other.num_rows
            )));
        }

        let mut result = Self::new(self.num_rows);
        for i in 0..self.bitmap.len() {
            result.bitmap[i] = self.bitmap[i] & other.bitmap[i];
        }
        result.recount();
        Ok(result)
    }

    /// OR two selection vectors.
    pub fn or(&self, other: &SelectionVector) -> Result<SelectionVector> {
        if self.num_rows != other.num_rows {
            return Err(Error::DimensionMismatch(format!(
                "Selection vector sizes don't match: {} vs {}",
                self.num_rows, other.num_rows
            )));
        }

        let mut result = Self::new(self.num_rows);
        for i in 0..self.bitmap.len() {
            result.bitmap[i] = self.bitmap[i] | other.bitmap[i];
        }
        result.recount();
        Ok(result)
    }

    /// NOT selection vector.
    pub fn not(&self) -> SelectionVector {
        let mut result = Self::new(self.num_rows);
        for i in 0..self.bitmap.len() {
            result.bitmap[i] = !self.bitmap[i];
        }

        // Clear extra bits in last word
        let extra_bits = self.num_rows % 64;
        if extra_bits > 0 && !result.bitmap.is_empty() {
            let last = result.bitmap.len() - 1;
            result.bitmap[last] &= (1u64 << extra_bits) - 1;
        }

        result.recount();
        result
    }

    /// Get indices of selected rows.
    pub fn selected_indices(&self) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.selected_count);
        for i in 0..self.num_rows {
            if self.is_selected(i) {
                indices.push(i);
            }
        }
        indices
    }
}

/// Vectorized filter executor.
pub struct VectorizedFilter;

impl VectorizedFilter {
    /// Filter f32 column with comparison.
    pub fn filter_f32(
        data: &[f32],
        op: FilterOp,
        threshold: f32,
    ) -> SelectionVector {
        let mut selection = SelectionVector::new(data.len());
        let num_words = selection.num_words();

        for word_idx in 0..num_words {
            let start = word_idx * 64;
            let end = (start + 64).min(data.len());
            let mut word = 0u64;

            for i in start..end {
                let bit = i - start;
                let pass = Self::compare_f32(data[i], op, threshold);
                if pass {
                    word |= 1u64 << bit;
                }
            }

            selection.set_word(word_idx, word);
        }

        selection.recount();
        selection
    }

    /// Filter f64 column with comparison.
    pub fn filter_f64(
        data: &[f64],
        op: FilterOp,
        threshold: f64,
    ) -> SelectionVector {
        let mut selection = SelectionVector::new(data.len());
        let num_words = selection.num_words();

        for word_idx in 0..num_words {
            let start = word_idx * 64;
            let end = (start + 64).min(data.len());
            let mut word = 0u64;

            for i in start..end {
                let bit = i - start;
                let pass = Self::compare_f64(data[i], op, threshold);
                if pass {
                    word |= 1u64 << bit;
                }
            }

            selection.set_word(word_idx, word);
        }

        selection.recount();
        selection
    }

    /// Filter i64 column with comparison.
    pub fn filter_i64(
        data: &[i64],
        op: FilterOp,
        threshold: i64,
    ) -> SelectionVector {
        let mut selection = SelectionVector::new(data.len());
        let num_words = selection.num_words();

        for word_idx in 0..num_words {
            let start = word_idx * 64;
            let end = (start + 64).min(data.len());
            let mut word = 0u64;

            for i in start..end {
                let bit = i - start;
                let pass = Self::compare_i64(data[i], op, threshold);
                if pass {
                    word |= 1u64 << bit;
                }
            }

            selection.set_word(word_idx, word);
        }

        selection.recount();
        selection
    }

    /// Filter i32 column with comparison.
    pub fn filter_i32(
        data: &[i32],
        op: FilterOp,
        threshold: i32,
    ) -> SelectionVector {
        let mut selection = SelectionVector::new(data.len());
        let num_words = selection.num_words();

        for word_idx in 0..num_words {
            let start = word_idx * 64;
            let end = (start + 64).min(data.len());
            let mut word = 0u64;

            for i in start..end {
                let bit = i - start;
                let pass = Self::compare_i32(data[i], op, threshold);
                if pass {
                    word |= 1u64 << bit;
                }
            }

            selection.set_word(word_idx, word);
        }

        selection.recount();
        selection
    }

    /// Compare f32 value.
    #[inline]
    fn compare_f32(value: f32, op: FilterOp, threshold: f32) -> bool {
        match op {
            FilterOp::Eq => value == threshold,
            FilterOp::Ne => value != threshold,
            FilterOp::Gt => value > threshold,
            FilterOp::Ge => value >= threshold,
            FilterOp::Lt => value < threshold,
            FilterOp::Le => value <= threshold,
        }
    }

    /// Compare f64 value.
    #[inline]
    fn compare_f64(value: f64, op: FilterOp, threshold: f64) -> bool {
        match op {
            FilterOp::Eq => value == threshold,
            FilterOp::Ne => value != threshold,
            FilterOp::Gt => value > threshold,
            FilterOp::Ge => value >= threshold,
            FilterOp::Lt => value < threshold,
            FilterOp::Le => value <= threshold,
        }
    }

    /// Compare i64 value.
    #[inline]
    fn compare_i64(value: i64, op: FilterOp, threshold: i64) -> bool {
        match op {
            FilterOp::Eq => value == threshold,
            FilterOp::Ne => value != threshold,
            FilterOp::Gt => value > threshold,
            FilterOp::Ge => value >= threshold,
            FilterOp::Lt => value < threshold,
            FilterOp::Le => value <= threshold,
        }
    }

    /// Compare i32 value.
    #[inline]
    fn compare_i32(value: i32, op: FilterOp, threshold: i32) -> bool {
        match op {
            FilterOp::Eq => value == threshold,
            FilterOp::Ne => value != threshold,
            FilterOp::Gt => value > threshold,
            FilterOp::Ge => value >= threshold,
            FilterOp::Lt => value < threshold,
            FilterOp::Le => value <= threshold,
        }
    }

    /// Apply filter to column.
    pub fn filter_column(column: &Column, predicate: &FilterPredicate) -> Result<SelectionVector> {
        match predicate {
            FilterPredicate::Int32(op, threshold) => {
                let data = column.as_i32()?;
                Ok(Self::filter_i32(data, *op, *threshold))
            }
            FilterPredicate::Int64(op, threshold) => {
                let data = column.as_i64()?;
                Ok(Self::filter_i64(data, *op, *threshold))
            }
            FilterPredicate::Float32(op, threshold) => {
                let data = column.as_f32()?;
                Ok(Self::filter_f32(data, *op, *threshold))
            }
            FilterPredicate::Float64(op, threshold) => {
                let data = column.as_f64()?;
                Ok(Self::filter_f64(data, *op, *threshold))
            }
        }
    }

    /// Compact column based on selection.
    pub fn compact_f32(data: &[f32], selection: &SelectionVector) -> Result<AlignedVec<f32>> {
        let mut result = AlignedVec::with_capacity(selection.count())?;

        for i in 0..data.len() {
            if selection.is_selected(i) {
                result.push(data[i])?;
            }
        }

        Ok(result)
    }

    /// Compact column based on selection.
    pub fn compact_f64(data: &[f64], selection: &SelectionVector) -> Result<AlignedVec<f64>> {
        let mut result = AlignedVec::with_capacity(selection.count())?;

        for i in 0..data.len() {
            if selection.is_selected(i) {
                result.push(data[i])?;
            }
        }

        Ok(result)
    }

    /// Compact column based on selection.
    pub fn compact_i64(data: &[i64], selection: &SelectionVector) -> Result<AlignedVec<i64>> {
        let mut result = AlignedVec::with_capacity(selection.count())?;

        for i in 0..data.len() {
            if selection.is_selected(i) {
                result.push(data[i])?;
            }
        }

        Ok(result)
    }
}

/// Range filter for BETWEEN queries.
pub struct RangeFilter;

impl RangeFilter {
    /// Filter values in range [low, high].
    pub fn filter_f32_range(data: &[f32], low: f32, high: f32) -> SelectionVector {
        let gt_low = VectorizedFilter::filter_f32(data, FilterOp::Ge, low);
        let lt_high = VectorizedFilter::filter_f32(data, FilterOp::Le, high);
        gt_low.and(&lt_high).unwrap()
    }

    /// Filter values in range [low, high].
    pub fn filter_f64_range(data: &[f64], low: f64, high: f64) -> SelectionVector {
        let gt_low = VectorizedFilter::filter_f64(data, FilterOp::Ge, low);
        let lt_high = VectorizedFilter::filter_f64(data, FilterOp::Le, high);
        gt_low.and(&lt_high).unwrap()
    }

    /// Filter values in range [low, high].
    pub fn filter_i64_range(data: &[i64], low: i64, high: i64) -> SelectionVector {
        let gt_low = VectorizedFilter::filter_i64(data, FilterOp::Ge, low);
        let lt_high = VectorizedFilter::filter_i64(data, FilterOp::Le, high);
        gt_low.and(&lt_high).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selection_vector() {
        let mut sel = SelectionVector::new(100);
        sel.set(10, true);
        sel.set(50, true);
        sel.recount();

        assert!(sel.is_selected(10));
        assert!(sel.is_selected(50));
        assert!(!sel.is_selected(20));
        assert_eq!(sel.count(), 2);
    }

    #[test]
    fn test_filter_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let selection = VectorizedFilter::filter_f32(&data, FilterOp::Gt, 5.0);

        assert!(!selection.is_selected(4)); // 5.0 not > 5.0
        assert!(selection.is_selected(5));  // 6.0 > 5.0
        assert_eq!(selection.count(), 5);
    }

    #[test]
    fn test_filter_i64() {
        let data = vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let selection = VectorizedFilter::filter_i64(&data, FilterOp::Le, 5);

        assert!(selection.is_selected(0));
        assert!(selection.is_selected(4));
        assert!(!selection.is_selected(5));
        assert_eq!(selection.count(), 5);
    }

    #[test]
    fn test_range_filter() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let selection = RangeFilter::filter_f32_range(&data, 3.0, 7.0);

        assert!(!selection.is_selected(1)); // 2.0 not in [3, 7]
        assert!(selection.is_selected(2));  // 3.0 in [3, 7]
        assert!(selection.is_selected(6));  // 7.0 in [3, 7]
        assert!(!selection.is_selected(7)); // 8.0 not in [3, 7]
        assert_eq!(selection.count(), 5);
    }

    #[test]
    fn test_selection_and_or() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let sel1 = VectorizedFilter::filter_f32(&data, FilterOp::Gt, 3.0);
        let sel2 = VectorizedFilter::filter_f32(&data, FilterOp::Lt, 6.0);

        let and_sel = sel1.and(&sel2).unwrap();
        assert_eq!(and_sel.count(), 2); // 4.0, 5.0

        let or_sel = sel1.or(&sel2).unwrap();
        assert_eq!(or_sel.count(), 8); // (>3 OR <6) covers all 8 elements
    }

    #[test]
    fn test_compact() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut selection = SelectionVector::new(5);
        selection.set(1, true);
        selection.set(3, true);
        selection.recount();

        let compacted = VectorizedFilter::compact_f32(&data, &selection).unwrap();
        assert_eq!(compacted.len(), 2);
        assert_eq!(compacted.get(0).unwrap(), 2.0);
        assert_eq!(compacted.get(1).unwrap(), 4.0);
    }
}
