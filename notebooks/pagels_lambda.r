# Pagel's lambda analysis - 3 methods comparison

cat("=== COMPREHENSIVE PAGEL'S LAMBDA ANALYSIS ===\n")
cat("Three methods will be compared for each trait:\n")
cat("1. GEIGER - Maximum Likelihood estimation\n")
cat("2. PHYTOOLS - Alternative ML with significance testing\n") 
cat("3. NLME - Phylogenetic Generalized Least Squares\n\n")

library(ape)
library(geiger)
library(phytools)
library(nlme)

tree <- read.tree("tree/tree_raxml.newick")
traits <- read.csv("data/life_history_traits_pca.csv")

# Check if tree is binary and resolve polytomies if needed
if(!is.binary(tree)) {
  cat("Warning: Tree is not binary. Resolving polytomies...\n")
  tree <- multi2di(tree)
}

rownames(traits) <- traits[,1]  # Use first column as row names
traits <- traits[,-1]  # Remove species name column

cat("Tree loaded with", length(tree$tip.label), "species\n")
cat("Trait data loaded with", nrow(traits), "species and", ncol(traits), "traits\n")
cat("Available columns:", paste(colnames(traits), collapse = ", "), "\n")

common_species <- intersect(tree$tip.label, rownames(traits))
cat("Common species found:", length(common_species), "\n")

tree <- drop.tip(tree, setdiff(tree$tip.label, common_species))
traits <- traits[common_species, ]

# Find common species and clean data
common_species <- intersect(tree$tip.label, rownames(traits))
cat("Common species found:", length(common_species), "\n")

tree_clean <- drop.tip(tree, setdiff(tree$tip.label, common_species))
traits_clean <- traits[common_species, ]
cat("After filtering:", length(tree$tip.label), "species in analysis\n")

# ========================================
# FUNCTION TO CALCULATE LAMBDA WITH 3 METHODS
# ========================================

calculate_lambda_three_methods <- function(trait_data, tree, trait_name) {
  cat("=== Analyzing", trait_name, "===\n")
  
  # Remove missing values
  valid_data <- !is.na(trait_data)
  trait_clean <- trait_data[valid_data]
  species_to_drop <- names(trait_data)[!valid_data]
  
  if(length(species_to_drop) > 0) {
    tree_trait <- drop.tip(tree, species_to_drop)
  } else {
    tree_trait <- tree
  }
  
  # Ensure tree is binary after tip removal
  if(!is.binary(tree_trait)) {
    tree_trait <- multi2di(tree_trait)
  }
  
  # Ensure data and tree match
  trait_final <- trait_clean[tree_trait$tip.label]
  
  n_species <- length(tree_trait$tip.label)
  cat("Species in analysis:", n_species, "\n")
  
  # Initialize results
  results <- list(
    geiger_lambda = NA,
    phytools_lambda = NA,
    nlme_lambda = NA
  )
  
  # METHOD 1: GEIGER
  cat("- GEIGER: ")
  tryCatch({
    result_geiger <- fitContinuous(tree_trait, trait_final, model = "lambda")
    results$geiger_lambda <- result_geiger$opt$lambda
    cat("Success (λ =", round(results$geiger_lambda, 4), ")\n")
  }, error = function(e) {
    cat("Failed -", e$message, "\n")
  })
  
  # METHOD 2: PHYTOOLS
  cat("- PHYTOOLS: ")
  tryCatch({
    result_phytools <- phylosig(tree_trait, trait_final, method = "lambda", test = TRUE)
    results$phytools_lambda <- result_phytools$lambda
    cat("Success (λ =", round(results$phytools_lambda, 4), " )\n")
  }, error = function(e) {
    cat("Failed -", e$message, "\n")
  })
  
  # METHOD 3: NLME
  cat("- NLME: ")
  tryCatch({
    analysis_data <- data.frame(
      species = names(trait_final),
      trait_value = trait_final,
      stringsAsFactors = FALSE
    )
    rownames(analysis_data) <- analysis_data$species
    
    result_nlme <- gls(trait_value ~ 1, 
                       data = analysis_data,
                       correlation = corPagel(value = 1, phy = tree_trait, form = ~species),
                       method = "ML")
    
    results$nlme_lambda <- result_nlme$modelStruct$corStruct[[1]]
    cat("Success (λ =", round(results$nlme_lambda, 4), ")\n")
  }, error = function(e) {
    cat("Failed -", e$message, "\n")
  })
  
  cat("\n")
  
  # Return results as data frame
  return(data.frame(
    Trait = trait_name,
    N = n_species,
    GEIGER_Lambda = results$geiger_lambda,
    PHYTOOLS_Lambda = results$phytools_lambda,
    NLME_Lambda = results$nlme_lambda,
    stringsAsFactors = FALSE
  ))
}

# ========================================
# ANALYSE ALL TRAITS
# ========================================

cat("=== ANALYZING ALL TRAITS ===\n\n")

# List of traits to analyse
trait_columns <- c("PC1", "Lb", "Li", "Lim", "Lp", "Lpm", "Ri", "Wwb", "Wwi", "Wwim", "Wwp", "ab", "am", "tp")
existing_traits <- trait_columns[trait_columns %in% colnames(traits_clean)]

cat("Traits to analyse:", paste(existing_traits, collapse = ", "), "\n\n")

# Initialize results list
all_results <- list()

# Analyse each trait
for(trait in existing_traits) {
  trait_data <- traits_clean[[trait]]
  names(trait_data) <- rownames(traits_clean)
  
  result <- calculate_lambda_three_methods(trait_data, tree_clean, trait)
  all_results[[trait]] <- result
}

# ========================================
# COMBINE RESULTS INTO FINAL TABLE
# ========================================

cat("=== COMPILING FINAL RESULTS ===\n")

# Combine all results
final_table <- do.call(rbind, all_results)
rownames(final_table) <- NULL

# Round numeric columns for display
numeric_cols <- c("GEIGER_Lambda", "PHYTOOLS_Lambda", "NLME_Lambda")
for(col in numeric_cols) {
  if(col %in% colnames(final_table)) {
    final_table[[col]] <- round(final_table[[col]], 6)
  }
}

# Display results
cat("\n=== COMPREHENSIVE LAMBDA COMPARISON TABLE ===\n\n")
print(final_table)

# save in csv 
write.csv(final_table, "lambda_results_raxml.csv", row.names = FALSE)
