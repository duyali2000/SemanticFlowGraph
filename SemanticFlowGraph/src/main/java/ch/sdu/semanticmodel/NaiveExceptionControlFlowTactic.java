package ch.sdu.semanticmodel;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Stack;

import org.apache.commons.lang3.tuple.Pair;

import spoon.reflect.code.CtCatch;
import spoon.reflect.code.CtCatchVariable;
import spoon.reflect.code.CtThrow;
import spoon.reflect.code.CtTry;
import spoon.reflect.code.CtTryWithResource;
import spoon.reflect.declaration.CtElement;

/**
 * A naive over-approximating model of exception control flow with limited support for finalizers.
 *
 * The model uses the following assumptions:
 *
 *   1) Any statement can potentially throw any exception.
 *   2) All exceptions thrown inside a try block are caught by the catchers immediately associated with the block.
 *
 * Support for finalizers is limited by the lack of modeling for the semantics of return statements in regards to
 * executing finalizers before actually returning. Because of this limitation, by default the implementation will
 * refuse to model the flow of a try-(catch-)finally construct that contains return statements. An option is
 * available to allow the model to produce a partially incorrect graph where return statements jump directly to the
 * exit without executing finalizers.
 */
public class NaiveExceptionControlFlowTactic implements ExceptionControlFlowTactic {
	/**
	 * Per-instance option flags for NaiveExceptionControlFlowStrategy.
	 */
	public enum Options {
		/**
		 * Add paths between the end of an empty try {} block and its catchers.
		 *
		 * Default: disabled.
		 *
		 * This option exists because expressions of the form "try { } catch(Exception e) { foo(); }" (i.e empty try
		 * blocks) are legal in Java, despite the statement "foo()" trivially being unreachable. In some use cases,
		 * excluding such unreachable statements from the control flow graph may be desirable, while in other cases the
		 * information loss may be undesirable. The default choice of not adding these paths was chosen due to how the
		 * produced graph more accurately models the actual control flow of an execution. Enabling the option produces
		 * a graph that can be said to show what the Java compiler considers to be reachable code.
		 */
		AddPathsForEmptyTryBlocks,

		/**
		 * Model (incorrectly) return statements as jumping directly to the exit node without executing any "in-scope"
		 * finalizers.
		 *
		 * Default: disabled.
		 *
		 * This option exists to provide a limited form of support for return statements in try-(catch-)finally
		 * constructs despite the lack of complete modeling for the semantics of return statements when finalizers are
		 * present. Depending on the use case, the incorrect aspects of the produced graph may be an acceptable
		 * tradeoff versus having return statements be completely unsupported when finalizers are used.
		 */
		ReturnWithoutFinalizers
	}

	public NaiveExceptionControlFlowTactic() {
		this(EnumSet.noneOf(Options.class));
	}

	public NaiveExceptionControlFlowTactic(EnumSet<Options> options) {
		instanceOptions = options;
		catchNodeStack = new Stack<>();
	}

	@Override
	public void handleTryStatement(ExtendedCDGBuilder builder, CtTry tryBlock, int tryindex) {
		
		if (!instanceOptions.contains(Options.ReturnWithoutFinalizers) && tryBlock.getFinalizer() != null) {
			// we temporally do nothing
		}

		ExtendedCodeDependencyGraph graph = builder.getResult();
		DependencyFlowNode lastNodeToUse = builder.getLastNodeToUse();
		ExtendedCDGBuilder.outlevel++;
		
		DependencyFlowNode tryNode = new DependencyFlowNode(null, graph, NodeKind.BRANCH,
				NodeType.TRY, NodeRole.BLOCKDEFAULT, null, tryindex, ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		graph.addVertex(tryNode);
		graph.addEdge(lastNodeToUse, tryNode);
		
		if(tryBlock instanceof CtTryWithResource) {
		   CtTryWithResource resource = (CtTryWithResource) tryBlock;
		   if (resource.getResources() != null && !resource.getResources().isEmpty()) {
//			 builder.printList(resource.getResources(),
//				null, "(",  ";",  ")", r -> builder.scan(r));
			 builder.printList(resource.getResources(),
						null, "",  ";",  "", r -> builder.scan(r));
		  }
		}
        
		builder.scan(tryBlock.getBody());
		DependencyFlowNode lastBodyNode = builder.getLastNodeToUse();
		DependencyFlowNode finallyNode = null;

		if (tryBlock.getFinalizer() != null) {
			finallyNode = new DependencyFlowNode(null, graph, NodeKind.BRANCH, 
					NodeType.FINALLY, NodeRole.BLOCKDEFAULT, null, graph.getTokenSequence().size(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
			graph.addVertex(finallyNode);
		}
		
		DependencyFlowNode convergeNode = new DependencyFlowNode(null, graph, NodeKind.BRANCH,
				NodeType.TryCONVERGE, NodeRole.BLOCKDEFAULT, null, graph.getTokenSequence().size(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		graph.addVertex(convergeNode);
		graph.addEdge(lastBodyNode, finallyNode != null ? finallyNode : convergeNode);

		for (CtCatch catchBlock : tryBlock.getCatchers()) {
			ExtendedCDGBuilder.innerindex++;
			graph.getTokenSequence().add(Pair.of("catch",null));
			DependencyFlowNode catchNode = new DependencyFlowNode(catchBlock.getParameter(), graph, NodeKind.BRANCH, 
					NodeType.CATCH, NodeRole.BLOCKDEFAULT, null,graph.getTokenSequence().size(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
			graph.addVertex(catchNode);

		//	graph.getTokenSequence().add(Pair.of("(",null));
			ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(graph.getTokenSequence());
			CtCatchVariable<? extends Throwable> parameter = catchBlock.getParameter();
			if (parameter != null && parameter.getMultiTypes().size() > 1) {
//				builder.printList(parameter.getMultiTypes(),
//						null, null, "|", null, type -> builder.scan(type));
				builder.printList(parameter.getMultiTypes(),
						null, null, "", null, type -> builder.scan(type));
				graph.getTokenSequence().add(Pair.of(parameter.getSimpleName(),parameter));
			} else {
				builder.scan(parameter);
			}
			ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(graph.getTokenSequence());
	        ArrayList<Pair<CtElement, Integer>> involvedVariables = builder.getInvolvedVariables(copybefore, copyafter);
	        builder.dealVariablesInGraph(involvedVariables,"others", null, catchNode, false);
	     //   graph.getTokenSequence().add(Pair.of(")",null));
	        
	        int beforeID=DependencyFlowNode.count;
			builder.scan(catchBlock.getBody());
	        int afterID=DependencyFlowNode.count;
	        if(afterID>beforeID)
			    graph.addEdge(catchNode, graph.findNodeById(beforeID+1));
			graph.addEdge(builder.getLastNodeToUse(), finallyNode != null ? finallyNode : convergeNode);		
		}

		if (finallyNode != null) {
			ExtendedCDGBuilder.innerindex++;
			graph.getTokenSequence().add(Pair.of("finally",null));
			finallyNode.setTokenIndex(graph.getTokenSequence().size());
	        int beforeID=DependencyFlowNode.count;
			builder.scan(tryBlock.getFinalizer());
	        int afterID=DependencyFlowNode.count;
            if(afterID>beforeID)
			     graph.addEdge(finallyNode, graph.findNodeById(beforeID+1));
			graph.addEdge(builder.getLastNodeToUse(), convergeNode);
		}
		
		convergeNode.setTokenIndex(graph.getTokenSequence().size());
		ExtendedCDGBuilder.outlevel--;
		builder.lastConverengenceNode = convergeNode;
	}

	@Override
	public void handleThrowStatement(ExtendedCDGBuilder builder, CtThrow throwStatement, int indexForThrow) {
		ExtendedCodeDependencyGraph graph = builder.getResult();
		DependencyFlowNode lastNodeToUse = builder.getLastNodeToUse();
		DependencyFlowNode throwNode = new DependencyFlowNode(throwStatement, graph, 
				NodeKind.BRANCH, NodeType.Exceptionthrow, NodeRole.BLOCKDEFAULT, null, indexForThrow, ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		graph.addVertex(throwNode);
		graph.addEdge(lastNodeToUse, throwNode);
	}

	@Override
	public void postProcess(ExtendedCodeDependencyGraph graph) {
	
	}

	/**
	 * Stack of catch nodes that statements parented by a try block may jump to.
	 */
	@SuppressWarnings("unused")
	private Stack<List<DependencyFlowNode>> catchNodeStack;

	/**
	 * Flag indicating whether paths should be added between an empty try {} block and its catchers.
	 */
	private EnumSet<Options> instanceOptions;
}

