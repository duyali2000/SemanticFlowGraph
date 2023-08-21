package ch.sdu.semanticmodel;

import org.jgrapht.graph.DefaultEdge;

@SuppressWarnings("serial")
public class DependencyFlowEdge extends DefaultEdge {
	
	boolean isBackEdge = false;

	public boolean isBackEdge() {
		return isBackEdge;
	}

	public void setBackEdge(boolean isLooopingEdge) {
		this.isBackEdge = isLooopingEdge;
	}

	public DependencyFlowNode getTargetNode() {
		return (DependencyFlowNode) getTarget();
	}

	public DependencyFlowNode getSourceNode() {
		return (DependencyFlowNode) getSource();
	}

}
