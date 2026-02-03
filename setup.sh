#!/bin/bash

# Supply Chain Domino Analyzer - Setup Script
# This script helps you set up the project

echo "=================================================="
echo "🔗 Supply Chain Domino Analyzer - Setup"
echo "=================================================="
echo ""

# Function to print colored output
print_success() {
    echo "✅ $1"
}

print_error() {
    echo "❌ $1"
}

print_info() {
    echo "ℹ️  $1"
}

# Check Python version
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is available
if command -v pip &> /dev/null || command -v pip3 &> /dev/null; then
    print_success "pip found"
else
    print_error "pip not found. Please install pip."
    exit 1
fi

echo ""
echo "=================================================="
echo "Select setup option:"
echo "=================================================="
echo "1. Quick Demo (Just open dashboard - no installation)"
echo "2. Full Setup (Install dependencies and train model)"
echo "3. API Server Only (Install minimal dependencies)"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        print_info "Quick Demo Mode Selected"
        print_info "Opening enhanced dashboard..."
        echo ""
        echo "The dashboard file is located at:"
        echo "  dashboard/enhanced_dashboard.html"
        echo ""
        echo "To use it:"
        echo "  1. Open the file in your web browser"
        echo "  2. Try the predefined scenarios"
        echo "  3. Run Monte Carlo simulations"
        echo "  4. Analyze portfolio risk"
        echo ""
        print_success "No installation needed - it works offline!"
        ;;
    
    2)
        echo ""
        print_info "Full Setup Mode Selected"
        echo ""
        
        # Create directories
        print_info "Creating project directories..."
        mkdir -p models notebooks
        print_success "Directories created"
        
        # Install dependencies
        print_info "Installing dependencies..."
        echo ""
        
        read -p "Do you want to install PyTorch with CUDA support? (y/n): " cuda_choice
        
        if [ "$cuda_choice" = "y" ]; then
            print_info "Installing PyTorch with CUDA..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --break-system-packages
        else
            print_info "Installing PyTorch (CPU only)..."
            pip install torch torchvision torchaudio --break-system-packages
        fi
        
        print_info "Installing other dependencies..."
        pip install torch-geometric --break-system-packages
        pip install networkx pandas numpy matplotlib seaborn plotly --break-system-packages
        pip install fastapi uvicorn pydantic --break-system-packages
        pip install scikit-learn --break-system-packages
        
        print_success "Dependencies installed"
        echo ""
        
        # Train model
        read -p "Do you want to train the model now? (y/n): " train_choice
        
        if [ "$train_choice" = "y" ]; then
            print_info "Training GNN model (this may take 5-10 minutes)..."
            cd scripts
            python3 train.py
            cd ..
            print_success "Model trained successfully"
        else
            print_info "Skipping model training"
            print_info "You can train later by running: cd scripts && python train.py"
        fi
        
        echo ""
        print_success "Full setup complete!"
        echo ""
        echo "Next steps:"
        echo "  1. Open dashboard/enhanced_dashboard.html in your browser"
        echo "  2. Run stress tests: cd scripts && python stress_test.py"
        echo "  3. Start API server: cd src && python api_server.py"
        ;;
    
    3)
        echo ""
        print_info "API Server Setup Mode Selected"
        echo ""
        
        print_info "Installing API dependencies..."
        pip install fastapi uvicorn pydantic --break-system-packages
        pip install torch pandas numpy --break-system-packages
        
        print_success "API dependencies installed"
        echo ""
        
        print_info "To start the API server:"
        echo "  cd src && python api_server.py"
        echo ""
        print_info "API will be available at:"
        echo "  http://localhost:8000"
        echo "  http://localhost:8000/docs (Swagger UI)"
        ;;
    
    *)
        print_error "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "=================================================="
print_success "Setup completed!"
echo "=================================================="
echo ""
echo "📚 For detailed instructions, see QUICKSTART.md"
echo "🎯 To get started immediately, open: dashboard/enhanced_dashboard.html"
echo ""