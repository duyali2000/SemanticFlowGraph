package ch.sdu.semanticmodel;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.HashMap;

public class GraphPrettyPrinter {

	private final ExtendedCodeDependencyGraph graph;

	private String filePathBase; 
	
	public GraphPrettyPrinter (ExtendedCodeDependencyGraph graph, String path) {
		this.graph = graph;
		filePathBase = path;
	}

	public void print(int fileindex) {

		try {
		    File fout = new File(this.filePathBase+String.valueOf(fileindex)+".txt");
		    FileOutputStream fos = new FileOutputStream(fout);
		    BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
	       
		    bw.write("Toke Sequence for code:");
		    bw.newLine();
		    for (int index = 0; index < graph.getTokenSequence().size(); index++) {
			    bw.write(graph.getTokenSequence().get(index).getLeft());
			    bw.newLine();
		    }
		    
		    bw.write("Graph Nodes and Related Information:");
		    bw.newLine();
		    int i = 0;
			HashMap<DependencyFlowNode, Integer> nodeIds = new HashMap<DependencyFlowNode, Integer>();
			for (DependencyFlowNode n : graph.vertexSet()) {
				i++;
				StringBuilder sb = new StringBuilder("Node ");
				bw.write(printNode(i, n, sb));
				nodeIds.put(n, i);
				bw.newLine();
			}
		    
		    bw.write("Graph Edges: ");
		    bw.newLine();
		    int j = 0;
			for (DependencyFlowEdge e : graph.edgeSet()) {
				String edge = nodeIds.get(e.getSourceNode()) + " -> " + nodeIds.get(e.getTargetNode());
				bw.write(edge);
				j++;
				if(j != (graph.edgeSet().size()))
				     bw.newLine();
			}
		    bw.close(); 
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
	}

	private String printNode(int i, DependencyFlowNode n, StringBuilder sb) {
		String labelStr = " [TokenIndex=";
		
		if(n.getKind()==NodeKind.VARIABLE) {
		    labelStr+= String.valueOf(n.getTokenIndex());
		} else {
			if(n.getType()==NodeType.ConditionalCONVERGE)
			  labelStr+= "---";
			else if(n.getType()==NodeType.DoInsideCONVERGE)
				  labelStr+= "---";
			else if(n.getType()==NodeType.DoOutsideCONVERGE)
				  labelStr+= "---";
			else if(n.getType()==NodeType.ForCONVERGE)
				  labelStr+= "---";
			else if(n.getType()==NodeType.ForEachCONVERGE)
				  labelStr+= "---";
			else if(n.getType()==NodeType.IfCONVERGE)
				  labelStr+= "---";
			else if(n.getType()==NodeType.SwitchCONVERGE)
				  labelStr+= "---";
			else if(n.getType()==NodeType.WhileCONVERGE)
				  labelStr+= "---";
			else if(n.getType()==NodeType.TryCONVERGE)
				  labelStr+= "---";
			else if(n.getType()==NodeType.BEGIN)
			  labelStr+= "---";
			else if(n.getType()==NodeType.EXIT)
			  labelStr+= "---";
			else if(n.getType()==NodeType.IfThen)
			  labelStr+= "---";
			else if(n.getType()==NodeType.Lambdaexit)
			  labelStr+= "---";
			else if(n.getType()==NodeType.Newclassexit)
			  labelStr+= "---";
			else labelStr+= String.valueOf(n.getTokenIndex());
		}
		labelStr+=",";
		
		labelStr+="Type=";
		labelStr+=n.getType().name();
		labelStr+=",";

		labelStr+="Role=";
		labelStr+=n.getRole().name();
		labelStr+=",";
		
		labelStr+="Kind=";
		labelStr+=n.getKind().name();
		labelStr+=",";
		
		labelStr+="Name=";
		if(n.getVariable()!=null)
		  labelStr+=n.getVariable().getShortName();
		else labelStr+= "---";
		labelStr+="]";
		
		sb.append(i).append(":").append(labelStr);
		return sb.toString();
	}
}
