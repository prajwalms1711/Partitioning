import re
import random
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import numpy as np

class VerilogParser:
    """Parser for Verilog gate-level netlists"""
    
    def __init__(self):
        self.gates = []
        self.wires = set()
        self.inputs = set()
        self.outputs = set()
        self.gate_types = ['and', 'or', 'not', 'nand', 'nor', 'xor', 'xnor', 'buf']
    
    def parse_file(self, filename):
        """Parse Verilog file and extract circuit information"""
        try:
            with open(filename, 'r') as file:
                content = file.read()
            return self.parse_content(content)
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return None
    
    def parse_content(self, content):
        """Parse Verilog content string"""
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            # Remove comments
            line = line.split('//')[0].strip()
            
            # Parse module declaration
            if line.startswith('module'):
                self._parse_module_declaration(line)
            
            # Parse wire declarations
            elif line.startswith('wire'):
                self._parse_wire_declaration(line)
            
            # Parse gate instances
            elif any(line.strip().startswith(gate) for gate in self.gate_types):
                self._parse_gate_instance(line)
        
        return {
            'gates': self.gates,
            'wires': list(self.wires),
            'inputs': list(self.inputs),
            'outputs': list(self.outputs)
        }
    
    def _parse_module_declaration(self, line):
        """Parse module declaration to extract I/O ports"""
        match = re.search(r'module\s+\w+\s*\((.*?)\)', line)
        if match:
            ports = match.group(1)
            
            # Extract inputs
            input_match = re.findall(r'input\s+([^,)]+)', ports)
            for inp in input_match:
                self.inputs.update(inp.split())
            
            # Extract outputs
            output_match = re.findall(r'output\s+([^,)]+)', ports)
            for out in output_match:
                self.outputs.update(out.split())
    
    def _parse_wire_declaration(self, line):
        """Parse wire declarations"""
        wire_match = re.search(r'wire\s+([^;]+)', line)
        if wire_match:
            wires = wire_match.group(1).split(',')
            for wire in wires:
                self.wires.add(wire.strip())
    
    def _parse_gate_instance(self, line):
        """Parse gate instance declarations"""
        pattern = r'(\w+)\s+(\w+)\s*\(([^)]+)\)\s*;'
        match = re.search(pattern, line)
        
        if match:
            gate_type = match.group(1)
            gate_name = match.group(2)
            connections = match.group(3)
            
            pins = [pin.strip() for pin in connections.split(',')]
            output_pin = pins[0] if pins else None
            input_pins = pins[1:] if len(pins) > 1 else []
            
            gate_info = {
                'name': gate_name,
                'type': gate_type,
                'output': output_pin,
                'inputs': input_pins,
                'all_pins': pins
            }
            
            self.gates.append(gate_info)
            
            for pin in pins:
                if pin not in self.inputs and pin not in self.outputs:
                    self.wires.add(pin)

class CircuitGraph:
    """Graph representation of the circuit"""
    
    def __init__(self, parsed_data):
        self.gates = parsed_data['gates']
        self.wires = parsed_data['wires']
        self.inputs = parsed_data['inputs']
        self.outputs = parsed_data['outputs']
        self.graph = nx.Graph()
        self._build_graph()
    
    def _build_graph(self):
        """Build graph from parsed circuit data"""
        for gate in self.gates:
            self.graph.add_node(gate['name'], 
                              gate_type=gate['type'],
                              output=gate['output'],
                              inputs=gate['inputs'])
        self._add_edges_by_nets()
    
    def _add_edges_by_nets(self):
        """Add edges between gates that share nets"""
        net_to_gates = defaultdict(list)
        
        for gate in self.gates:
            for pin in gate['all_pins']:
                net_to_gates[pin].append(gate['name'])
        
        for net, gate_list in net_to_gates.items():
            for i in range(len(gate_list)):
                for j in range(i + 1, len(gate_list)):
                    gate1, gate2 = gate_list[i], gate_list[j]
                    if self.graph.has_edge(gate1, gate2):
                        self.graph[gate1][gate2]['weight'] += 1
                    else:
                        self.graph.add_edge(gate1, gate2, weight=1, net=net)
    
    def get_nodes(self):
        return list(self.graph.nodes())
    
    def get_edges(self):
        return list(self.graph.edges(data=True))
    
    def get_neighbors(self, node):
        return list(self.graph.neighbors(node))
    
    def get_edge_weight(self, node1, node2):
        if self.graph.has_edge(node1, node2):
            return self.graph[node1][node2].get('weight', 1)
        return 0

class KLPartitioner:
    """Kernighan-Lin Algorithm Implementation"""
    
    def __init__(self, circuit_graph):
        self.circuit_graph = circuit_graph
        self.nodes = list(circuit_graph.graph.nodes())
        self.num_nodes = len(self.nodes)
        
        if self.num_nodes < 2:
            raise ValueError("Graph must have at least 2 nodes for partitioning")
        
        self.partition_a = set()
        self.partition_b = set()
        self.max_iterations = 20
        self.improvement_threshold = 0
    
    def initial_partition(self):
        """Create initial balanced partition"""
        shuffled_nodes = self.nodes.copy()
        random.shuffle(shuffled_nodes)
        mid = self.num_nodes // 2
        self.partition_a = set(shuffled_nodes[:mid])
        self.partition_b = set(shuffled_nodes[mid:])
    
    def calculate_cut_cost(self, partition_a=None, partition_b=None):
        """Calculate the cut cost between partitions"""
        if partition_a is None:
            partition_a = self.partition_a
        if partition_b is None:
            partition_b = self.partition_b
        
        cut_cost = 0
        for edge in self.circuit_graph.get_edges():
            node1, node2, data = edge
            if (node1 in partition_a and node2 in partition_b) or \
               (node1 in partition_b and node2 in partition_a):
                cut_cost += data.get('weight', 1)
        return cut_cost
    
    def calculate_gain(self, node):
        """Calculate gain for moving a node to the other partition"""
        external_cost = 0
        internal_cost = 0
        
        node_partition = self.partition_a if node in self.partition_a else self.partition_b
        other_partition = self.partition_b if node in self.partition_a else self.partition_a
        
        for neighbor in self.circuit_graph.get_neighbors(node):
            weight = self.circuit_graph.get_edge_weight(node, neighbor)
            if neighbor in other_partition:
                external_cost += weight
            elif neighbor in node_partition:
                internal_cost += weight
        
        return external_cost - internal_cost
    
    def find_best_swap_pair(self):
        """Find the best pair of nodes to swap between partitions"""
        best_gain = float('-inf')
        best_pair = None
        
        for node_a in self.partition_a:
            for node_b in self.partition_b:
                gain_a = self.calculate_gain(node_a)
                gain_b = self.calculate_gain(node_b)
                edge_weight = self.circuit_graph.get_edge_weight(node_a, node_b)
                combined_gain = gain_a + gain_b - 2 * edge_weight
                
                if combined_gain > best_gain:
                    best_gain = combined_gain
                    best_pair = (node_a, node_b)
        
        return best_pair, best_gain
    
    def perform_swap(self, node_a, node_b):
        """Swap two nodes between partitions"""
        self.partition_a.remove(node_a)
        self.partition_a.add(node_b)
        self.partition_b.remove(node_b)
        self.partition_b.add(node_a)
    
    def partition(self):
        """Main KL partitioning algorithm"""
        self.initial_partition()
        initial_cost = self.calculate_cut_cost()
        best_cost = initial_cost
        best_partition_a = self.partition_a.copy()
        best_partition_b = self.partition_b.copy()
        
        iteration = 0
        improvements = []
        
        while iteration < self.max_iterations:
            iteration += 1
            swap_pair, gain = self.find_best_swap_pair()
            
            if swap_pair is None or gain <= self.improvement_threshold:
                break
            
            node_a, node_b = swap_pair
            self.perform_swap(node_a, node_b)
            new_cost = self.calculate_cut_cost()
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_partition_a = self.partition_a.copy()
                best_partition_b = self.partition_b.copy()
            
            improvements.append({
                'iteration': iteration,
                'swap': f"{node_a} ↔ {node_b}",
                'gain': gain,
                'cost': new_cost
            })
        
        self.partition_a = best_partition_a
        self.partition_b = best_partition_b
        
        return {
            'partition_a': list(self.partition_a),
            'partition_b': list(self.partition_b),
            'cut_cost': best_cost,
            'initial_cost': initial_cost,
            'improvement': initial_cost - best_cost,
            'iterations': iteration,
            'history': improvements
        }

class Visualizer:
    """Visualization of partitioned circuit"""
    
    def __init__(self, circuit_graph, partition_result):
        self.circuit_graph = circuit_graph
        self.partition_result = partition_result
    
    def draw_partition(self):
        """Draw the partitioned circuit"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        self._draw_original_circuit(axes[0])
        self._draw_partitioned_circuit(axes[1])
        plt.tight_layout()
        return fig
    
    def _draw_original_circuit(self, ax):
        """Draw original circuit graph"""
        G = self.circuit_graph.graph
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=500, alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.5, width=1.5)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')
        ax.set_title("Original Circuit Graph", fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def _draw_partitioned_circuit(self, ax):
        """Draw partitioned circuit with different colors and highlighted cut edges"""
        G = self.circuit_graph.graph
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        partition_a = set(self.partition_result['partition_a'])
        partition_b = set(self.partition_result['partition_b'])
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=list(partition_a),
                             node_color='lightcoral', node_size=500, alpha=0.8,
                             ax=ax, label=f'Partition A ({len(partition_a)} nodes)')
        nx.draw_networkx_nodes(G, pos, nodelist=list(partition_b),
                             node_color='lightgreen', node_size=500, alpha=0.8,
                             ax=ax, label=f'Partition B ({len(partition_b)} nodes)')
        
        # Draw edges
        cut_edges = []
        internal_edges = []
        for edge in G.edges():
            node1, node2 = edge
            if (node1 in partition_a and node2 in partition_b) or \
               (node1 in partition_b and node2 in partition_a):
                cut_edges.append(edge)
            else:
                internal_edges.append(edge)
        
        nx.draw_networkx_edges(G, pos, edgelist=internal_edges,
                             edge_color='gray', alpha=0.3, width=1.0, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=cut_edges,
                             edge_color='red', width=3.0, alpha=0.8, ax=ax,
                             label=f'Cut edges ({len(cut_edges)})')
        
        # Draw labels and legend
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')
        ax.set_title(f"Partitioned Circuit (Cut Cost: {self.partition_result['cut_cost']})", 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.axis('off')
