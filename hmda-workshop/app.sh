#!/bin/bash
# app.sh - Optimized HMDA Dashboard launcher for Domino

echo "=============================================="
echo "    HMDA Loan Analytics Dashboard Launcher    "
echo "=============================================="
echo ""

# Set environment variables for better performance
export R_MAX_NUM_DLLS=500
export OMP_NUM_THREADS=1

# Print system information for debugging
echo "System Information:"
echo "-------------------"
echo "R Version:"
R --version | head -n 1
echo "Working Directory: $(pwd)"
echo "Available Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "CPU Cores: $(nproc)"
echo ""

# Check if model files exist
echo "Checking required model files:"
echo "-----------------------------"
for file in "models/best_model.rds" "models/preprocessing_config.rds" "models/factor_levels.rds" "models/test_data.rds"; do
    if [ -f "$file" ]; then
        echo "✓ Found: $file ($(du -h "$file" | cut -f1))"
    else
        echo "✗ Missing: $file"
    fi
done
echo ""

# Install and verify packages
echo "Installing required R packages:"
echo "------------------------------"

R --vanilla --slave << 'EOF'
# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# Define required packages (INCLUDING shinyjs for overlay fixes)
packages_to_install <- c(
  "shiny",
  "shinydashboard", 
  "shinydashboardPlus",
  "shinyWidgets",
  "shinyjs",          # ADDED for JavaScript overlay removal
  "tidyverse",
  "DT",
  "plotly",
  "randomForest",
  "glmnet",
  "scales"
)

# Function to install packages with error handling
install_package_safe <- function(pkg) {
  tryCatch({
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat(paste("Installing", pkg, "...\n"))
      install.packages(pkg, quiet = TRUE, dependencies = TRUE)
      
      # Verify installation
      if (requireNamespace(pkg, quietly = TRUE)) {
        cat(paste("✓", pkg, "installed successfully\n"))
        return(TRUE)
      } else {
        cat(paste("✗", pkg, "installation failed\n"))
        return(FALSE)
      }
    } else {
      cat(paste("✓", pkg, "already installed\n"))
      return(TRUE)
    }
  }, error = function(e) {
    cat(paste("✗ Error installing", pkg, ":", e$message, "\n"))
    return(FALSE)
  })
}

# Install packages
failed_packages <- character()
for (pkg in packages_to_install) {
  if (!install_package_safe(pkg)) {
    failed_packages <- c(failed_packages, pkg)
  }
}

# Report results
if (length(failed_packages) > 0) {
  cat("\n⚠ WARNING: The following packages failed to install:\n")
  cat(paste("-", failed_packages, collapse = "\n"), "\n")
  cat("\nThe dashboard may not function properly.\n")
} else {
  cat("\n✓ All packages installed successfully!\n")
}

# Load and verify critical packages
cat("\nVerifying package loads:\n")
cat("------------------------\n")
critical_packages <- c("shiny", "shinydashboard", "shinyjs", "plotly", "DT")
for (pkg in critical_packages) {
  tryCatch({
    library(pkg, character.only = TRUE)
    cat(paste("✓", pkg, "loaded successfully\n"))
  }, error = function(e) {
    cat(paste("✗", pkg, "failed to load:", e$message, "\n"))
  })
}

cat("\nPackage setup complete.\n")
EOF

echo ""
echo "Starting HMDA Dashboard..."
echo "=========================="
echo "Dashboard URL: http://0.0.0.0:8888"
echo "Press Ctrl+C to stop the server"
echo ""

# Create a simple test to verify Shiny can start
R --vanilla --slave << 'EOF'
# Test if Shiny can create a basic app
tryCatch({
  library(shiny)
  library(shinyjs)
  cat("✓ Shiny framework is functional\n")
}, error = function(e) {
  cat("✗ Shiny framework error:", e$message, "\n")
  quit(status = 1)
})
EOF

# Run the dashboard with error handling
R --vanilla --slave << 'EOF'
tryCatch({
  # Set options for Domino deployment
  options(
    shiny.port = 8888,
    shiny.host = '0.0.0.0',
    shiny.launch.browser = FALSE,
    shiny.error = function() {
      cat("\n✗ Shiny application error occurred\n")
    }
  )
  
  # Check if app file exists - UPDATE THIS IF YOUR FILE HAS A DIFFERENT NAME
  app_file <- 'hmda_dashboard.R'  # Change to 'app.R' if you saved it with that name
  if (!file.exists(app_file)) {
    # Try alternative names
    if (file.exists('app.R')) {
      app_file <- 'app.R'
      cat("Using app.R instead of hmda_dashboard.R\n")
    } else {
      stop(paste("Dashboard file not found. Looked for:", app_file, "and app.R"))
    }
  }
  
  cat(paste("\nLaunching dashboard from:", app_file, "\n"))
  cat("========================================\n\n")
  
  # Run the app
  shiny::runApp(app_file, launch.browser = FALSE)
  
}, error = function(e) {
  cat("\n✗ Failed to start dashboard:\n")
  cat(paste("  Error:", e$message, "\n"))
  cat("\nTroubleshooting tips:\n")
  cat("1. Ensure hmda_dashboard.R or app.R exists in the current directory\n")
  cat("2. Check that all model files are present in the models/ directory\n")
  cat("3. Verify R packages are installed correctly\n")
  cat("4. Check Domino logs for more details\n")
  quit(status = 1)
})
EOF