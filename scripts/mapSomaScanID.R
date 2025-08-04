library(SomaScan.db)
library(AnnotationDbi)

#' Map SomaScan Probe IDs to Gene Symbols
#'
#' Converts SomaScan probe identifiers (e.g., "seq.10000.28") to human-readable
#' gene symbols using the SomaScan.db annotation package. Probe IDs that cannot
#' be mapped are retained in their original format.
#'
#' @param input_file Path to input CSV file with SomaScan data
#' @param output_file Path to output CSV file with mapped gene symbols
#' @return Invisibly returns the modified data frame
#'
#' @details
#' The function expects a CSV file where:
#' - First column contains sample identifiers
#' - Remaining columns contain SomaScan probe data with names like "seq.10000.28"
#' 
#' Probe ID transformation:
#' 1. Removes "seq." prefix from column names
#' 2. Converts remaining dots to hyphens (e.g., "10000.28" -> "10000-28")
#' 3. Maps to gene symbols using SomaScan.db annotation
#' 4. Retains original probe ID if mapping fails
#'
#' @examples
#' map_somascan_ids("input.csv", "output_with_symbols.csv")
map_somascan_ids <- function(input_file, output_file) {
  if (!file.exists(input_file)) {
    stop("Input file does not exist: ", input_file)
  }
  
  df <- read.csv(input_file, stringsAsFactors = FALSE, check.names = FALSE)
  
  if (ncol(df) < 2) {
    stop("Input file must have at least 2 columns (sample ID + probe data)")
  }
  
  orig_cols <- colnames(df)
  sample_col <- orig_cols[1]
  probe_cols <- orig_cols[-1]
  
  probe_ids <- sub("^seq\\.", "", probe_cols)
  probe_ids <- gsub("\\.", "-", probe_ids)
  
  symbol_map <- mapIds(SomaScan.db,
                       keys = probe_ids,
                       column = "SYMBOL",
                       keytype = "PROBEID",
                       multiVals = "first")
  
  missing <- is.na(symbol_map)
  symbol_map[missing] <- probe_ids[missing]
  
  new_cols <- c(sample_col, unname(symbol_map))
  colnames(df) <- new_cols
  
  write.csv(df, output_file, row.names = FALSE)
  
  n_mapped <- sum(!missing)
  n_total <- length(probe_ids)
  message("Mapped ", n_mapped, " of ", n_total, " probes to gene symbols")
  message("Output written to: ", output_file)
  
  invisible(df)
}

input_file <- "/Users/sky2333/Desktop/somascan_overlap_test_imputed_knn.csv"
output_file <- "/Users/sky2333/Desktop/somascan_overlap_test_imputed_knn_symbol.csv"

map_somascan_ids(input_file, output_file)