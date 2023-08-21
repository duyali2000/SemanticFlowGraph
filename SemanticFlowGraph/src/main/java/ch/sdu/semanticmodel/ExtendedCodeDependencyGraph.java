package ch.sdu.semanticmodel;

import java.util.ArrayList;
import java.util.List;
import org.apache.commons.lang3.tuple.Pair;
import org.jgrapht.graph.DefaultDirectedGraph;

import spoon.reflect.declaration.CtElement;

@SuppressWarnings("serial")
public class ExtendedCodeDependencyGraph extends DefaultDirectedGraph<DependencyFlowNode, DependencyFlowEdge> {
    	
    ArrayList<Pair<String, CtElement>> tokensequence = new ArrayList<Pair<String, CtElement>>(); 
    
    public ExtendedCodeDependencyGraph(Class<? extends DependencyFlowEdge> edgeClass) {
		super(edgeClass);
	}

	public ExtendedCodeDependencyGraph () {
		super(DependencyFlowEdge.class);
	}
	
	public ArrayList<Pair<String, CtElement>> getTokenSequence() {
		return this.tokensequence;
	}
	
	public void clearTokenSequence() {
		this.tokensequence.clear();
	}
		
	public DependencyFlowNode findNodeById(int id) {
		for (DependencyFlowNode n : vertexSet()) {
			if (n.getId() == id) {
				return n;
			}
		}
		return null;
	}
	
	@Override
	public DependencyFlowEdge addEdge(DependencyFlowNode source, DependencyFlowNode target) {
		if (!containsVertex(source)) {
			addVertex(source);
		}
		if (!containsVertex(target)) {
			addVertex(target);
		}
		return super.addEdge(source, target);
	}
	
	public List<DependencyFlowNode> findNodesOfType(NodeType type) {
		ArrayList<DependencyFlowNode> result = new ArrayList<DependencyFlowNode>();
		for (DependencyFlowNode n : vertexSet()) {
			if (n.getType().equals(type)) {
				result.add(n);
			}
		}
		return result;
	}
	
	public List<DependencyFlowNode> findNodesOfKind(NodeKind kind) {
		ArrayList<DependencyFlowNode> result = new ArrayList<DependencyFlowNode>();
		for (DependencyFlowNode n : vertexSet()) {
			if (n.getKind().equals(kind)) {
				result.add(n);
			}
		}
		return result;
	}
	
	private DependencyFlowNode exitNode;
	
	private void simplify(NodeType type) {
//		try {
//			List<DependencyFlowNode> convergence = findNodesOfType(type);
//			for (DependencyFlowNode n : convergence) {
//				Set<DependencyFlowEdge> incoming = incomingEdgesOf(n);
//				Set<DependencyFlowEdge> outgoing = outgoingEdgesOf(n);
//				if (incoming != null && outgoing != null) {
//					for (DependencyFlowEdge in : incoming) {
//						for (DependencyFlowEdge out : outgoing) {
//							DependencyFlowEdge ed = addEdge(in.getSourceNode(), out.getTargetNode());
//							if (ed != null) {
//								ed.setBackEdge(out.isBackEdge() || in.isBackEdge());
//							}
//						}
//					}
//				}
//
//				for (DependencyFlowEdge e : edgesOf(n)) {
//					removeEdge(e);
//				}
//				removeVertex(n);
//			}
//		} catch (Exception e) {
//			throw e;
//		}
//		//Clean the exit node
//		exitNode = null;
	}
	
	public void toGraphVisText(int fileindex, String path) {
		GraphPrettyPrinter p = new GraphPrettyPrinter(this, path);
		p.print(fileindex);
	}

	public void simplify() {
		// simplify(NodeType.CONVERGE);
	}
	
	public DependencyFlowNode getExitNode() {
		if (exitNode == null) {
			exitNode = findNodesOfType(NodeType.EXIT).get(0);
		}
		return exitNode;
	}
}
