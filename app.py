from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
from vlsi_tools import VerilogParser, CircuitGraph, KLPartitioner, Visualizer
from io import BytesIO
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'verilog_file' not in request.files:
            return redirect(request.url)
        
        file = request.files['verilog_file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the file
            result = process_verilog_file(filepath)
            
            # Generate visualization
            plot_url = generate_visualization(result['circuit_graph'], result['partition_result'])
            
            return render_template('index.html', 
                                 plot_url=plot_url,
                                 result=result['partition_result'],
                                 filename=filename)
    
    return render_template('index.html')

def process_verilog_file(filepath):
    """Process the uploaded Verilog file and return partitioning results"""
    # Step 1: Parse Verilog
    print(f"\nStep 1: Parsing Verilog netlist from '{filepath}'...")
    parser = VerilogParser()
    parsed_data = parser.parse_file(filepath)
    
    if parsed_data is None:
        return None
    
    # Step 2: Build circuit graph
    print(f"\nStep 2: Building circuit graph...")
    circuit_graph = CircuitGraph(parsed_data)
    
    # Step 3: Run KL algorithm
    print(f"\nStep 3: Running KL partitioning algorithm...")
    partitioner = KLPartitioner(circuit_graph)
    partition_result = partitioner.partition()
    
    return {
        'circuit_graph': circuit_graph,
        'partition_result': partition_result
    }

def generate_visualization(circuit_graph, partition_result):
    """Generate visualization and return as base64 encoded image"""
    # Create visualization
    visualizer = Visualizer(circuit_graph, partition_result)
    
    # Save plot to a bytes buffer
    buffer = BytesIO()
    fig = visualizer.draw_partition()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    plt.close(fig)
    
    # Encode the image
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{image_base64}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
