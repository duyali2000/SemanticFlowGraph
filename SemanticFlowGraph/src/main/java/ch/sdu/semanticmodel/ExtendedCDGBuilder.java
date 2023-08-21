package ch.sdu.semanticmodel;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.lang.annotation.Annotation;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Stack;
import java.util.function.Consumer;

import org.apache.commons.lang3.tuple.Pair;

import spoon.SpoonException;
import spoon.reflect.code.BinaryOperatorKind;
import spoon.reflect.code.CaseKind;
import spoon.reflect.code.CtAbstractInvocation;
import spoon.reflect.code.CtAnnotationFieldAccess;
import spoon.reflect.code.CtArrayAccess;
import spoon.reflect.code.CtArrayRead;
import spoon.reflect.code.CtArrayWrite;
import spoon.reflect.code.CtAssert;
import spoon.reflect.code.CtAssignment;
import spoon.reflect.code.CtBinaryOperator;
import spoon.reflect.code.CtBlock;
import spoon.reflect.code.CtBreak;
import spoon.reflect.code.CtCase;
import spoon.reflect.code.CtCatch;
import spoon.reflect.code.CtCatchVariable;
import spoon.reflect.code.CtCodeSnippetExpression;
import spoon.reflect.code.CtCodeSnippetStatement;
import spoon.reflect.code.CtComment;
import spoon.reflect.code.CtConditional;
import spoon.reflect.code.CtConstructorCall;
import spoon.reflect.code.CtContinue;
import spoon.reflect.code.CtDo;
import spoon.reflect.code.CtExecutableReferenceExpression;
import spoon.reflect.code.CtExpression;
import spoon.reflect.code.CtFieldAccess;
import spoon.reflect.code.CtFieldRead;
import spoon.reflect.code.CtFieldWrite;
import spoon.reflect.code.CtFor;
import spoon.reflect.code.CtForEach;
import spoon.reflect.code.CtIf;
import spoon.reflect.code.CtInvocation;
import spoon.reflect.code.CtJavaDoc;
import spoon.reflect.code.CtJavaDocTag;
import spoon.reflect.code.CtLambda;
import spoon.reflect.code.CtLiteral;
import spoon.reflect.code.CtLocalVariable;
import spoon.reflect.code.CtNewArray;
import spoon.reflect.code.CtNewClass;
import spoon.reflect.code.CtOperatorAssignment;
import spoon.reflect.code.CtResource;
import spoon.reflect.code.CtReturn;
import spoon.reflect.code.CtStatement;
import spoon.reflect.code.CtStatementList;
import spoon.reflect.code.CtSuperAccess;
import spoon.reflect.code.CtSwitch;
import spoon.reflect.code.CtSwitchExpression;
import spoon.reflect.code.CtSynchronized;
import spoon.reflect.code.CtTargetedExpression;
import spoon.reflect.code.CtTextBlock;
import spoon.reflect.code.CtThisAccess;
import spoon.reflect.code.CtThrow;
import spoon.reflect.code.CtTry;
import spoon.reflect.code.CtTryWithResource;
import spoon.reflect.code.CtTypeAccess;
import spoon.reflect.code.CtTypePattern;
import spoon.reflect.code.CtUnaryOperator;
import spoon.reflect.code.CtVariableRead;
import spoon.reflect.code.CtVariableWrite;
import spoon.reflect.code.CtWhile;
import spoon.reflect.code.CtYieldStatement;
import spoon.reflect.code.UnaryOperatorKind;
import spoon.reflect.declaration.CtAnnotation;
import spoon.reflect.declaration.CtAnnotationMethod;
import spoon.reflect.declaration.CtAnnotationType;
import spoon.reflect.declaration.CtAnonymousExecutable;
import spoon.reflect.declaration.CtClass;
import spoon.reflect.declaration.CtCompilationUnit;
import spoon.reflect.declaration.CtConstructor;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.declaration.CtEnum;
import spoon.reflect.declaration.CtEnumValue;
import spoon.reflect.declaration.CtExecutable;
import spoon.reflect.declaration.CtField;
import spoon.reflect.declaration.CtFormalTypeDeclarer;
import spoon.reflect.declaration.CtImport;
import spoon.reflect.declaration.CtInterface;
import spoon.reflect.declaration.CtMethod;
import spoon.reflect.declaration.CtModifiable;
import spoon.reflect.declaration.CtModule;
import spoon.reflect.declaration.CtModuleRequirement;
import spoon.reflect.declaration.CtNamedElement;
import spoon.reflect.declaration.CtPackage;
import spoon.reflect.declaration.CtPackageDeclaration;
import spoon.reflect.declaration.CtPackageExport;
import spoon.reflect.declaration.CtParameter;
import spoon.reflect.declaration.CtProvidedService;
import spoon.reflect.declaration.CtRecord;
import spoon.reflect.declaration.CtRecordComponent;
import spoon.reflect.declaration.CtType;
import spoon.reflect.declaration.CtTypeMember;
import spoon.reflect.declaration.CtTypeParameter;
import spoon.reflect.declaration.CtTypedElement;
import spoon.reflect.declaration.CtUsedService;
import spoon.reflect.declaration.CtVariable;
import spoon.reflect.declaration.ModifierKind;
import spoon.reflect.declaration.ParentNotInitializedException;
import spoon.reflect.factory.Factory;
import spoon.reflect.factory.TypeFactory;
import spoon.reflect.path.CtRole;
import spoon.reflect.reference.CtActualTypeContainer;
import spoon.reflect.reference.CtArrayTypeReference;
import spoon.reflect.reference.CtCatchVariableReference;
import spoon.reflect.reference.CtExecutableReference;
import spoon.reflect.reference.CtFieldReference;
import spoon.reflect.reference.CtIntersectionTypeReference;
import spoon.reflect.reference.CtLocalVariableReference;
import spoon.reflect.reference.CtModuleReference;
import spoon.reflect.reference.CtPackageReference;
import spoon.reflect.reference.CtParameterReference;
import spoon.reflect.reference.CtTypeMemberWildcardImportReference;
import spoon.reflect.reference.CtTypeParameterReference;
import spoon.reflect.reference.CtTypeReference;
import spoon.reflect.reference.CtUnboundVariableReference;
import spoon.reflect.reference.CtVariableReference;
import spoon.reflect.reference.CtWildcardReference;
import spoon.reflect.visitor.CtVisitor;
import spoon.reflect.visitor.filter.TypeFilter;
import spoon.support.reflect.CtExtendedModifier;

public class ExtendedCDGBuilder implements CtVisitor {
	
	boolean forceWildcardGenerics = true;
	boolean ignoreEnclosingClass = false;
	boolean ignoreImplicit = true;
	boolean ignoreGenerics = false;
	boolean SKIP_ARRAY = false;
	boolean FIRST_FOR_VARIABLE = false;
	boolean NEXT_FOR_VARIABLE = false;
	boolean IGNORE_STATIC_ACCESS= false;
	
	private boolean isFirst = true;	
	private boolean isExperAssignmentVars=false;
	Deque<CtExpression<?>> parenthesedExpression = new ArrayDeque<>();
	Deque<CtType<?>> currentThis = new ArrayDeque<>();
	protected ExtendedCodeDependencyGraph result = new ExtendedCodeDependencyGraph(DependencyFlowEdge.class);
	private CtType<?> currentTopLevel;
    static int outlevel=0;
    static int innerindex=0;	
	protected TypeFactory typeFactory;
	
	protected DependencyFlowNode exitNode = null; 
	protected DependencyFlowNode beginNode = null;
	
	protected ExceptionControlFlowTactic exceptionControlFlowTactic;
	//This stack pushes all the nodes to which a break statement may jump to.
	protected Stack<DependencyFlowNode> breakingBad = new Stack<>();
	//This stack pushes all the nodes to which a continue statement may jump to.
	protected Stack<DependencyFlowNode> continueBad = new Stack<>();
	protected HashMap<String, CtStatement> labeledStatement = new HashMap<>();

	protected Stack<DependencyFlowNode> lastControlNode = new Stack<>();
	protected DependencyFlowNode lastAssignedNode = null;

	protected DependencyFlowNode lastConverengenceNode = null;
	protected DependencyFlowNode lastlastnodetouse = null;
	protected ArrayList<Pair<CtElement, Integer>> variablesConditionalThen = new ArrayList<Pair<CtElement, Integer>>();
	protected ArrayList<Pair<CtElement, Integer>> variablesConditionalElse = new ArrayList<Pair<CtElement, Integer>>();
	protected ArrayList<Pair<CtElement, Integer>> arrayIndexVarsAssigned = new ArrayList<Pair<CtElement, Integer>>();
	protected ArrayList<DependencyFlowNode> lastdealedNodes = new ArrayList<DependencyFlowNode>();
	protected ArrayList<DependencyFlowNode> lastdealedNonEmptyNodes = new ArrayList<DependencyFlowNode>();

	public void setExceptionControlFlowTactic(ExceptionControlFlowTactic tactic) {
		exceptionControlFlowTactic = tactic;
	}
	
	public DependencyFlowNode getLastNode() {
		if(lastConverengenceNode!=null) {
			return lastConverengenceNode;
		}
		else {
			return result.findNodeById(DependencyFlowNode.count);
		}
	}

	public void setLastAssignedNode(DependencyFlowNode node) {
		lastAssignedNode = node;
	}
	
	public ExtendedCodeDependencyGraph getResult() {
		return this.result;
	}
	
	public DependencyFlowNode getLastNodeToUse() {	 
		if(this.getLastNode().getKind()==NodeKind.BRANCH)
			return this.getLastNode();
		else {
			if(!lastdealedNodes.isEmpty())	
			   return lastdealedNodes.get(0);
			else if(!lastdealedNonEmptyNodes.isEmpty())	
			   return lastdealedNonEmptyNodes.get(0);
			else return this.getLastNode();
		}
	}
	
	public ExtendedCodeDependencyGraph build(CtElement s, TypeFactory typefactory) {
		init();
		this.typeFactory = typefactory;
		s.accept(this);
		tryAddEdge(this.getLastNodeToUse(), exitNode);
		exitNode.setTokenIndex(result.getTokenSequence().size()); 
		return result;
	}
	
	protected void init() {
		DependencyFlowNode.count=0;
		ExtendedCDGBuilder.outlevel=0;
		ExtendedCDGBuilder.innerindex=0;
		exitNode = new DependencyFlowNode (null, result,
				NodeKind.BRANCH, NodeType.EXIT, NodeRole.BLOCKDEFAULT, null, 0, ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		beginNode = new DependencyFlowNode (null, result,
				NodeKind.BRANCH, NodeType.BEGIN, NodeRole.BLOCKDEFAULT, null, 0, ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(beginNode);
		result.addVertex(exitNode);
		
		forceWildcardGenerics = true;
		ignoreEnclosingClass = false;
		ignoreImplicit = true;
		ignoreGenerics = false;
		SKIP_ARRAY = false;
		FIRST_FOR_VARIABLE = false;
		NEXT_FOR_VARIABLE = false;
		IGNORE_STATIC_ACCESS= false;
		isFirst = true;	
		isExperAssignmentVars=false;
	}
	
	protected void tryAddEdge(DependencyFlowNode source, DependencyFlowNode target) {
		tryAddEdge(source, target, false, false);
	}

	protected void tryAddEdge(DependencyFlowNode source, DependencyFlowNode target, boolean isLooping, boolean breakDance) {
		boolean isBreak = source != null && source.getStatement() instanceof CtBreak;
		boolean isContinue = source != null && source.getStatement() instanceof CtContinue;

		if (source != null && target != null
			&& !result.containsEdge(source, target)
			&& (isLooping || breakDance || !(isBreak || isContinue))) {
			DependencyFlowEdge e = result.addEdge(source, target);
			e.setBackEdge(isLooping);
		}
	}
	
	public void pushCurrentThis(CtType<?> type) {
		currentThis.push(type);
	}

	public void popCurrentThis() {
		currentThis.pop();
	}
	
	protected void dealVariablesInGraph(ArrayList<Pair<CtElement, Integer>> involvedVariables, String mode, 
			DependencyFlowNode special, DependencyFlowNode lastStatExper, boolean considergetvaluefromassigned) {
		VariableUseType useType;
		if(mode.equals("Assigned")) 
			useType=VariableUseType.ASSIGNED;
		else if(mode.equals("Assignment"))
			useType=VariableUseType.ASSIGNMENT;
		else useType=VariableUseType.OTHERS;
		
		for(int index=0; index<involvedVariables.size();index++) {
        	Pair<CtElement, Integer> current=involvedVariables.get(index);
        	Pair<VariableType, String> varinfo = geVariableTypeAndShortname (current.getLeft());
        	CtElement element=current.getLeft();
        	Variable varstudy = new Variable (varinfo.getLeft(), useType, 
        			current.getRight(), varinfo.getRight(), element);
        	DependencyFlowNode existnode = findExistNodeForElement(current.getRight());
        	
        	if(useType == VariableUseType.ASSIGNED ) { 
        		if(existnode == null) {
        		    DependencyFlowNode assignedNode = new DependencyFlowNode (element, result,
        					NodeKind.VARIABLE, getNodeType(element), getNodeRole(element), varstudy, current.getRight(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
        		    result.addVertex(assignedNode);
        		    if(lastStatExper!=null && lastStatExper.getKind() == NodeKind.BRANCH && index==0 && lastStatExper.getType()!=NodeType.Branchbreak && lastStatExper.getType()!=NodeType.Branchcontinue) {
        			   tryAddEdge(lastStatExper, assignedNode);
        		    }
        		    this.setLastAssignedNode(assignedNode);
        		    lastdealedNodes.add(assignedNode);
        		} 
        	}
        	
            if(useType == VariableUseType.ASSIGNMENT) {  
        	   if(existnode == null) {
        		  DependencyFlowNode assignmentNode = new DependencyFlowNode (element, result,
        					NodeKind.VARIABLE, getNodeType(element), getNodeRole(element), varstudy, current.getRight(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
        		  result.addVertex(assignmentNode);
        		  tryAddEdge(assignmentNode, this.lastAssignedNode);	
    			  ArrayList<DependencyFlowNode> varfrom = getNodesValueFrom(assignmentNode,considergetvaluefromassigned);
    			  for(int innerindex=0; innerindex<varfrom.size(); innerindex++) 
        			 tryAddEdge(varfrom.get(innerindex), assignmentNode);		  
        	   } else {
         		  tryAddEdge(existnode, this.lastAssignedNode);
        	   }
        	}
           
            if(useType == VariableUseType.OTHERS) {  
        	   if(existnode == null) {
       		      DependencyFlowNode otherNode = new DependencyFlowNode (element, result,
       					NodeKind.VARIABLE, getNodeType(element), getNodeRole(element), varstudy, current.getRight(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
    		      result.addVertex(otherNode);
    		      if(!isExperAssignmentVars)
       		           lastdealedNodes.add(otherNode);
    		      if(lastStatExper!=null && lastStatExper.getKind() == NodeKind.BRANCH && index==0 && lastStatExper.getType()!=NodeType.Branchbreak && lastStatExper.getType()!=NodeType.Branchcontinue) {
    			     tryAddEdge(lastStatExper, otherNode);
    		      }
			      ArrayList<DependencyFlowNode> varfrom = getNodesValueFrom (otherNode, considergetvaluefromassigned);
			      for(int innerindex=0; innerindex<varfrom.size(); innerindex++) { 
    			      tryAddEdge(varfrom.get(innerindex), otherNode);	
			      }
			      if(special!=null) {
    			      tryAddEdge(otherNode, special);	
			      }
        	  } else {
        		  if(special!=null) {
    			      tryAddEdge(existnode, special);	
			      }
        	   }
       	   }
        }
	}
	
	protected ArrayList<DependencyFlowNode> getNodesValueFrom (DependencyFlowNode nodestudy, boolean considersameline) {
		ArrayList<DependencyFlowNode> identifiednodes = new ArrayList<DependencyFlowNode>();
		for(DependencyFlowNode node:this.result.vertexSet()) {
			if(nodestudy.getTokenIndex() != node.getTokenIndex() && node.getKind()==NodeKind.VARIABLE 
					&& nodestudy.getVariable().getShortName().equals(node.getVariable().getShortName())) {
				if(whetherVarsCompatiable(nodestudy,node)) {
					if(considersameline) {
						if(node!=this.lastAssignedNode)
							identifiednodes.add(node);
					} else identifiednodes.add(node);
				}
			}
		}
		
		if(identifiednodes.size() > 1)
		    return refinelist(identifiednodes);
		else return identifiednodes;
	}
		
	protected ArrayList<DependencyFlowNode> refinelist(ArrayList<DependencyFlowNode> identifiednodes) {
		ArrayList<DependencyFlowNode> refine = new ArrayList<DependencyFlowNode>();
		DependencyFlowNode nodeBiggestLine = retriveNodeBiggestLine (identifiednodes);
		refine.add(nodeBiggestLine);
		ArrayList<Integer> unique_innerindex = new ArrayList<Integer>();
		
		for(int index=0; index<identifiednodes.size(); index++) {
			if(identifiednodes.get(index).getOutLevel() == nodeBiggestLine.getOutLevel() && 
			   identifiednodes.get(index).getInnerIndex() != nodeBiggestLine.getInnerIndex()) {
				if(!unique_innerindex.contains(identifiednodes.get(index).getInnerIndex()))
					unique_innerindex.add(identifiednodes.get(index).getInnerIndex());
			}
		}
		
		for(int number=0; number<unique_innerindex.size(); number++) {
			ArrayList<DependencyFlowNode> currentstudy = new ArrayList<DependencyFlowNode>();
            int currentinnerindex=unique_innerindex.get(number);
            for(int index=0; index<identifiednodes.size(); index++) {
    			if(identifiednodes.get(index).getOutLevel() == nodeBiggestLine.getOutLevel() && 
    			   identifiednodes.get(index).getInnerIndex() == currentinnerindex) {
    				currentstudy.add(identifiednodes.get(index));
    			}
    		} 
            if(currentstudy != null)
               refine.add(retriveNodeBiggestLine(currentstudy));
		}
		return refine;
	}
	
	protected DependencyFlowNode retriveNodeBiggestLine(ArrayList<DependencyFlowNode> current) {
//		DependencyFlowNode nodeBiggestLine=null;
//		int currentline = current.get(0).getVariable().getAssociatedElement().getPosition().getLine();
//		nodeBiggestLine = current.get(0);	
//		for(int index=1; index<current.size(); index++) {
//			if(current.get(index).getVariable().getAssociatedElement().getPosition().getLine()>currentline) {
//				currentline = current.get(index).getVariable().getAssociatedElement().getPosition().getLine();
//				nodeBiggestLine = current.get(index);
//			}
//		}
//		return nodeBiggestLine;
		
		DependencyFlowNode nodeBiggestLine=null;
		int currenttokenindex = current.get(0).getTokenIndex();
		nodeBiggestLine = current.get(0);	
		for(int index=1; index<current.size(); index++) {
			if(current.get(index).getTokenIndex()>currenttokenindex) {
				   currenttokenindex = current.get(index).getTokenIndex();
				   nodeBiggestLine = current.get(index);
			}
		}
		return nodeBiggestLine;		
	}
	
	protected DependencyFlowNode findExistNodeForElement(int tokenindex) {		
        for(int index=DependencyFlowNode.count; index>=1; index--) {
        	DependencyFlowNode current = result.findNodeById(index);
        	if( current.getKind()==NodeKind.VARIABLE && current.getTokenIndex() == tokenindex) {
        		return current;
        	}
		} 
        return null;
	}
	
	protected boolean whetherVarsCompatiable (DependencyFlowNode nodestudy, DependencyFlowNode nodeingraph) {
		VariableType studyvarType = nodestudy.getVariable().getVariableType();
		VariableType graphnodevarType = nodeingraph.getVariable().getVariableType();
		
		if(nodeingraph.getVariable().getVariableUseType() != VariableUseType.ASSIGNED 
				&& ! (graphnodevarType==VariableType.PARAMETER|| graphnodevarType==VariableType.CatchVariable
				||graphnodevarType==VariableType.Resource || graphnodevarType==VariableType.Resource))
			return false;
		if(studyvarType==VariableType.PARAMETER|| studyvarType==VariableType.CatchVariable
				||studyvarType==VariableType.Resource || studyvarType==VariableType.Resource)
			return false;
		if(studyvarType==VariableType.LocalVariableReference && !(graphnodevarType==VariableType.LOCALEVARIABLE
				||graphnodevarType==VariableType.LocalVariableReference|| graphnodevarType==VariableType.Resource))
			return false;
		if(studyvarType==VariableType.ParameterReference && !(graphnodevarType==VariableType.PARAMETER
				||graphnodevarType==VariableType.ParameterReference))
			return false;
		if(studyvarType==VariableType.CatchVariableReference && !(graphnodevarType==VariableType.CatchVariable
				||graphnodevarType==VariableType.CatchVariableReference))
			return false;
		if(studyvarType==VariableType.FieldReference && !(graphnodevarType==VariableType.FIELD
				|| graphnodevarType==VariableType.FieldAccess || graphnodevarType==VariableType.FieldReference 
				|| graphnodevarType==VariableType.RecordComponent))
			return false;
		if(studyvarType==VariableType.UnboundVariableReference && graphnodevarType!=VariableType.UnboundVariableReference)
			return false;
       
        return true;    
	}
	
	protected Pair<VariableType, String> geVariableTypeAndShortname(CtElement element) {	
		if(element instanceof CtCatchVariable) 
			return Pair.of(VariableType.CatchVariable, ((CtCatchVariable<?>)element).getSimpleName());
		if(element instanceof CtLocalVariable) 
			return Pair.of(VariableType.LOCALEVARIABLE, ((CtLocalVariable<?>)element).getSimpleName());
		if(element instanceof CtField) 
			return Pair.of(VariableType.FIELD, ((CtField<?>)element).getSimpleName());
		if(element instanceof CtParameter) 
			return Pair.of(VariableType.PARAMETER, ((CtParameter<?>)element).getSimpleName());
		if(element instanceof CtRecordComponent) 
			return Pair.of(VariableType.RecordComponent, ((CtRecordComponent)element).getSimpleName());
		if(element instanceof CtResource) 
			return Pair.of(VariableType.Resource, ((CtResource<?>)element).getSimpleName());
		if(element instanceof CtCatchVariableReference) 
			return Pair.of(VariableType.CatchVariableReference, ((CtCatchVariableReference<?>)element).getSimpleName());
		if(element instanceof CtFieldReference) 
			return Pair.of(VariableType.FieldReference, ((CtFieldReference<?>)element).getSimpleName());
		if(element instanceof CtParameterReference) 
			return Pair.of(VariableType.ParameterReference, ((CtParameterReference<?>)element).getSimpleName());
		if(element instanceof CtLocalVariableReference) 
			return Pair.of(VariableType.LocalVariableReference, ((CtLocalVariableReference<?>)element).getSimpleName());
		if(element instanceof CtUnboundVariableReference) 
			return Pair.of(VariableType.UnboundVariableReference, ((CtUnboundVariableReference<?>)element).getSimpleName());
		if(element instanceof CtFieldAccess) 
			return Pair.of(VariableType.FieldAccess, ((CtFieldAccess<?>)element).getVariable().getSimpleName());
		if(element instanceof CtVariableRead) 
			return Pair.of(VariableType.VarRead, ((CtVariableRead<?>)element).getVariable().getSimpleName());
		if(element instanceof CtVariableWrite) 
			return Pair.of(VariableType.VarWrite, ((CtVariableWrite<?>)element).getVariable().getSimpleName());
		
		return Pair.of(VariableType.Unknown, element.getShortRepresentation());
	}
	
	protected ArrayList<Pair<CtElement, Integer>> getInvolvedVariables(ArrayList<Pair<String, CtElement>> before, 
			ArrayList<Pair<String, CtElement>> after) {
		ArrayList<Pair<CtElement, Integer>> diffVariables = new ArrayList<Pair<CtElement, Integer>>();
		for(int index=before.size(); index<after.size(); index++) {
			if(after.get(index).getRight()!=null)
				diffVariables.add(Pair.of(after.get(index).getRight(),index+1));
		}
		return diffVariables;
	}
	
	public NodeType getNodeType (CtElement variable) {
		CtTypeReference<?> type=null;	
		if(variable instanceof CtTypedElement) {
			type = ((CtTypedElement<?>)variable).getType();
		} else if(variable instanceof CtVariableReference) {
			type = ((CtVariableReference<?>)variable).getType();
		} else { }
		
		if(type!=null) {
			if(typeFactory.INTEGER_PRIMITIVE.equals(type) || typeFactory.INTEGER.equals(type))
				return NodeType.INTEGER;
			if(typeFactory.FLOAT_PRIMITIVE.equals(type) || typeFactory.FLOAT.equals(type))
				return NodeType.FLOAT;
			if(typeFactory.LONG_PRIMITIVE.equals(type) || typeFactory.LONG.equals(type))
				return NodeType.LONG;
			if(typeFactory.DOUBLE_PRIMITIVE.equals(type) || typeFactory.DOUBLE.equals(type))
				return NodeType.DOUBLE;
			if(typeFactory.VOID_PRIMITIVE.equals(type) || typeFactory.VOID.equals(type))
				return NodeType.VOID;
			if(typeFactory.BOOLEAN_PRIMITIVE.equals(type) || typeFactory.BOOLEAN.equals(type))
				return NodeType.BOOLEAN;
			if(typeFactory.BYTE_PRIMITIVE.equals(type) || typeFactory.BYTE.equals(type))
				return NodeType.BYTE;
			if(typeFactory.CHARACTER_PRIMITIVE.equals(type) || typeFactory.CHARACTER.equals(type))
				return NodeType.CHARACTER;
			if(typeFactory.SHORT_PRIMITIVE.equals(type) || typeFactory.SHORT.equals(type))
				return NodeType.SHORT;
			if(type.isPrimitive())
			    return NodeType.PrimitiveOthers;
			if(!type.isPrimitive()) {
				if(typeFactory.DATE.equals(type))
					return NodeType.DATE;
				if(typeFactory.ITERABLE.equals(type))
					return NodeType.ITERABLE;
				if(typeFactory.COLLECTION.equals(type))
					return NodeType.COLLECTION;
				if(typeFactory.LIST.equals(type))
					return NodeType.LIST;
				if(typeFactory.SET.equals(type))
					return NodeType.SET;
				if(typeFactory.MAP.equals(type))
					return NodeType.MAP;
				if(typeFactory.STRING.equals(type))
					return NodeType.STRING;
				if(contains(type.getQualifiedName()))
					return NodeType.OtherJDKType;
				return NodeType.UserDefinedType;
			}
		}
		return NodeType.NullType;
	}
	
	private static Set<String> CS = new HashSet<String>();
    static {
        try {
            File file = new File(System.getProperty("java.home"),
                    "lib/classlist");
            @SuppressWarnings("resource")
			BufferedReader r = new BufferedReader(new FileReader(file));
            String l;
            while (true) {
                l = r.readLine();
                if (l == null) {
                    break;
                } else {
                    CS.add(l.replace('/', '.'));
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public boolean contains(String o) {
        return CS.contains(o) || o.startsWith("java") || o.startsWith("com.sun")
                || o.startsWith("sun") || o.startsWith("oracle")
                || o.startsWith("org.xml") || o.startsWith("com.oracle");
    }
	
	public NodeRole getNodeRole(CtElement parameter) {
		if(parameter instanceof CtField) {
			return NodeRole.FieldIntr;
		} else if (parameter instanceof CtParameter) {
			return NodeRole.ParameterIntr;
		} else if(parameter instanceof CtLocalVariable) {
			return NodeRole.LocVarIntr;
		} else if(parameter instanceof CtCatchVariable) {
			return NodeRole.CatVarIntr;
		} else if(parameter instanceof CtResource) {
			return NodeRole.ResourceVarIntr;
		} else if(parameter instanceof CtRecordComponent) {
			return NodeRole.RecordComponentIntr;
		} else {
			if(parameter instanceof CtFieldReference) {
				parameter = parameter.getParent();
			} 
			
			if(parameter.getParent() instanceof CtAnnotation)
				return NodeRole.AnnotationValue;
			if(parameter.getParent() instanceof CtAssignment && parameter.getRoleInParent() == CtRole.ASSIGNED)
				return NodeRole.Assigned;
			if(parameter.getParent() instanceof CtAssignment && parameter.getRoleInParent() == CtRole.ASSIGNMENT)
				return NodeRole.Assignment;
			if(parameter.getParent() instanceof CtArrayAccess && parameter.getRoleInParent() == CtRole.TARGET)
				return NodeRole.ArrayTarget;
			if(parameter.getParent() instanceof CtArrayAccess && parameter.getRoleInParent() == CtRole.EXPRESSION)
				return NodeRole.ArrayExpression;
			if(parameter.getParent() instanceof CtAbstractInvocation && parameter.getRoleInParent() == CtRole.TARGET)
				return NodeRole.InvocationTarget;
			if(parameter.getParent() instanceof CtAbstractInvocation && parameter.getRoleInParent() == CtRole.ARGUMENT)
				return NodeRole.InvocationArgument;
			if(parameter.getParent() instanceof CtBinaryOperator && parameter.getRoleInParent() == CtRole.LEFT_OPERAND) {
				CtBinaryOperator<?> operator=(CtBinaryOperator<?>) parameter.getParent();
				if(operator.getKind()==BinaryOperatorKind.OR||operator.getKind()==BinaryOperatorKind.AND)
				    return NodeRole.RelationOperatorLeft;
				if(operator.getKind()==BinaryOperatorKind.BITOR||operator.getKind()==BinaryOperatorKind.BITXOR
						||operator.getKind()==BinaryOperatorKind.BITAND ||operator.getKind()==BinaryOperatorKind.SL
						||operator.getKind()==BinaryOperatorKind.SR ||operator.getKind()==BinaryOperatorKind.USR)
				    return NodeRole.BitOperatorLeft;
				if(operator.getKind()==BinaryOperatorKind.EQ||operator.getKind()==BinaryOperatorKind.NE
						||operator.getKind()==BinaryOperatorKind.LT||operator.getKind()==BinaryOperatorKind.GT
						||operator.getKind()==BinaryOperatorKind.GE||operator.getKind()==BinaryOperatorKind.LE)
				    return NodeRole.CompareOperatorLeft;
				if(operator.getKind()==BinaryOperatorKind.PLUS||operator.getKind()==BinaryOperatorKind.MINUS
						||operator.getKind()==BinaryOperatorKind.MUL||operator.getKind()==BinaryOperatorKind.DIV
						||operator.getKind()==BinaryOperatorKind.MOD)
				    return NodeRole.MathOperatorLeft;
				if(operator.getKind()==BinaryOperatorKind.INSTANCEOF)
				    return NodeRole.InstanceOperatorLeft;
			}
			
			if(parameter.getParent() instanceof CtBinaryOperator && parameter.getRoleInParent() == CtRole.RIGHT_OPERAND) {
				CtBinaryOperator<?> operator=(CtBinaryOperator<?>) parameter.getParent();
				if(operator.getKind()==BinaryOperatorKind.OR||operator.getKind()==BinaryOperatorKind.AND)
				    return NodeRole.RelationOperatorRight;
				if(operator.getKind()==BinaryOperatorKind.BITOR||operator.getKind()==BinaryOperatorKind.BITXOR
						||operator.getKind()==BinaryOperatorKind.BITAND ||operator.getKind()==BinaryOperatorKind.SL
						||operator.getKind()==BinaryOperatorKind.SR ||operator.getKind()==BinaryOperatorKind.USR)
				    return NodeRole.BitOperatorRight;				
				if(operator.getKind()==BinaryOperatorKind.EQ||operator.getKind()==BinaryOperatorKind.NE
						||operator.getKind()==BinaryOperatorKind.LT||operator.getKind()==BinaryOperatorKind.GT
						||operator.getKind()==BinaryOperatorKind.GE||operator.getKind()==BinaryOperatorKind.LE)
				    return NodeRole.CompareOperatorRight;
				if(operator.getKind()==BinaryOperatorKind.PLUS||operator.getKind()==BinaryOperatorKind.MINUS
						||operator.getKind()==BinaryOperatorKind.MUL||operator.getKind()==BinaryOperatorKind.DIV
						||operator.getKind()==BinaryOperatorKind.MOD)
				    return NodeRole.MathOperatorRight;
				if(operator.getKind()==BinaryOperatorKind.INSTANCEOF)
				    return NodeRole.InstanceOperatorRight;
			}
			
			if(parameter.getParent() instanceof CtUnaryOperator) {
				CtUnaryOperator<?> operator=(CtUnaryOperator<?>) parameter.getParent();
				if(operator.getKind()==UnaryOperatorKind.PREINC||operator.getKind()==UnaryOperatorKind.PREDEC
						|| operator.getKind()==UnaryOperatorKind.POSTINC||operator.getKind()==UnaryOperatorKind.POSTDEC)
				    return NodeRole.UnaryOperatorPrePostAssign;
				if(operator.getKind()==UnaryOperatorKind.POS||operator.getKind()==UnaryOperatorKind.NEG
						|| operator.getKind()==UnaryOperatorKind.NOT||operator.getKind()==UnaryOperatorKind.COMPL)
				    return NodeRole.UnaryOperatorOthers;
			}
			
			if(parameter.getParent() instanceof CtAssert)
				return NodeRole.AssertValue;
			if(parameter.getParent() instanceof CtSwitch || parameter.getParent() instanceof CtSwitchExpression)
				return NodeRole.SwitchSelector;
			if(parameter.getParent() instanceof CtCase)
			    return NodeRole.CaseValue;
			if(parameter.getParent() instanceof CtConditional && parameter.getRoleInParent() == CtRole.CONDITION)
				return NodeRole.ConditionalCond;
			if(parameter.getParent() instanceof CtConditional && parameter.getRoleInParent() == CtRole.THEN)
				return NodeRole.ConditionalThen;
			if(parameter.getParent() instanceof CtConditional && parameter.getRoleInParent() == CtRole.ELSE)
				return NodeRole.ConditionalElse;
			if(parameter.getParent() instanceof CtReturn)
				return NodeRole.ReturnExper;
			if(parameter.getParent() instanceof CtConstructorCall && parameter.getRoleInParent() == CtRole.ARGUMENT)
				return NodeRole.ConstructorCallArgument;
			if(parameter.getParent() instanceof CtDo && parameter.getRoleInParent() == CtRole.EXPRESSION)
				return NodeRole.DoWhileExper;
			if(parameter.getParent() instanceof CtWhile && parameter.getRoleInParent() == CtRole.EXPRESSION)
				return NodeRole.WhileExper;
			if(parameter.getParent() instanceof CtForEach && parameter.getRoleInParent() == CtRole.EXPRESSION)
				return NodeRole.ForEachExper;
			if(parameter.getParent() instanceof CtIf && parameter.getRoleInParent() == CtRole.CONDITION)
				return NodeRole.IfExper; 
			if(parameter.getParent().getParent() instanceof CtBlock && parameter.getParent().getParent().getRoleInParent() == CtRole.ELSE)
				return NodeRole.ElseExper; 
			if(parameter.getParent() instanceof CtYieldStatement)
				return NodeRole.YieldExper; 
			if(parameter.getParent() instanceof CtSynchronized && parameter.getRoleInParent() == CtRole.EXPRESSION)
				return NodeRole.SynchronizedExper; 
			if(parameter.getParent() instanceof CtFieldRead && parameter.getRoleInParent() == CtRole.TARGET)
				return NodeRole.FieldReadTarget;  //e.g., arr.length
			return NodeRole.VarUnknown;
		}
	}
	
	@Override
	public <A extends Annotation> void visitCtAnnotation(CtAnnotation<A> annotation) {
		DependencyFlowNode nodeFromLastStatement=this.getLastNode();
		this.writeAnnotations(annotation);
		result.getTokenSequence().add(Pair.of("@",null));
		scan(annotation.getAnnotationType());
		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		if (!annotation.getValues().isEmpty()) {
//			this.printList(annotation.getValues().entrySet(),
//				null, "(", ",", ")",
//				e -> {
//					if ((annotation.getValues().size() == 1 && "value".equals(e.getKey())) == false) {
//						result.getTokenSequence().add(Pair.of(e.getKey(),null));
//						result.getTokenSequence().add(Pair.of("=",null));
//					}
//					this.writeAnnotationElement(annotation.getFactory(), e.getValue());
//			 });
			this.printList(annotation.getValues().entrySet(),
					null, "", "", "",
					e -> {
						if ((annotation.getValues().size() == 1 && "value".equals(e.getKey())) == false) {
							result.getTokenSequence().add(Pair.of(e.getKey(),null));
							result.getTokenSequence().add(Pair.of("=",null));
						}
						this.writeAnnotationElement(annotation.getFactory(), e.getValue());
				 });
		}
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
        dealVariablesInGraph(involvedVariables,"others", null, nodeFromLastStatement, false);
	}
		
	public void writeAnnotationElement(Factory factory, Object value) {
		if (value instanceof CtTypeAccess) {
			this.scan((CtTypeAccess<?>) value);
			result.getTokenSequence().add(Pair.of(".class",null));
		} else if (value instanceof CtFieldReference) {
			this.scan(((CtFieldReference<?>) value).getDeclaringType());
		//	result.getTokenSequence().add(Pair.of(".",null));
			result.getTokenSequence().add(Pair.of(((CtFieldReference<?>) value).getSimpleName(), 
					(CtFieldReference<?>) value));
		} else if (value instanceof CtElement) {
			this.scan((CtElement) value);
		} else if (value instanceof String) {
			result.getTokenSequence().add(Pair.of("\"" + (String) value + "\"",null));
		} else if (value instanceof Collection) {
//			printList((Collection<?>) value, null,
//				 "{", ",", "}", obj -> writeAnnotationElement(factory, obj));
			printList((Collection<?>) value, null,
					 "", "", "", obj -> writeAnnotationElement(factory, obj));
		} else if (value instanceof Object[]) {
//			printList(Arrays.asList((Object[]) value), null,
//				 "{", ",", "}", obj -> writeAnnotationElement(factory, obj));
			printList(Arrays.asList((Object[]) value), null,
					 "", "", "", obj -> writeAnnotationElement(factory, obj));
		} else if (value instanceof Enum) {
			this.ignoreGenerics=true;
			this.scan(factory.Type().createReference(((Enum<?>) value).getDeclaringClass()));
			this.ignoreGenerics=false;
//			result.getTokenSequence().add(Pair.of("."+value.toString(),null));
			result.getTokenSequence().add(Pair.of(value.toString(),null));
		} else {
			result.getTokenSequence().add(Pair.of(value.toString(),null));
		}
	}
		
	public void writeAnnotations(CtElement element) {
		for (CtAnnotation<?> annotation : element.getAnnotations()) {
			// if element is a type reference and the parent is a typed element
			// which contains exactly the same annotation, then we are certainly in this case:
			// @myAnnotation String myField
			// in which case the annotation is attached to the type and the variable
			// in that case, we only print the annotation once.
			if (element.isParentInitialized() && element instanceof CtTypeReference && 
					(element.getParent() instanceof CtTypedElement) && 
					element.getParent().getAnnotations().contains(annotation)) {
					continue;
			}
			this.scan(annotation);
		}
	}
	
	@Override
	public <T> void visitCtTypeReference(CtTypeReference<T> ref) {
		visitCtTypeReference(ref, true);
	}
	
	public <T> void visitCtTypeReference(CtTypeReference<T> ref, boolean withGenerics) {
		if (!isPrintTypeReference(ref)) {
			return;
		}
		if (ref.isPrimitive()) {
			this.writeAnnotations(ref);
			result.getTokenSequence().add(Pair.of(ref.getSimpleName(), null));
			return;
		}
		boolean isInner = ref.getDeclaringType() != null;
		if (isInner) {
			if (!ignoreEnclosingClass && !ref.isLocalType()) {
				CtTypeReference<?> accessType = ref.getAccessType();
				if (!accessType.isAnonymous() && isPrintTypeReference(accessType)) {	
					if (!withGenerics) {
						this.ignoreGenerics = true;
					}
					scan(accessType);		
				//	result.getTokenSequence().add(Pair.of(".",null));
					this.ignoreGenerics = false;
				}
			}
			this.writeAnnotations(ref);
			if (ref.isLocalType()) {
				result.getTokenSequence().add(Pair.of(stripLeadingDigits(ref.getSimpleName()),null));
			} else {
				result.getTokenSequence().add(Pair.of(ref.getSimpleName(),null));
			}
		} else {
			if (ref.getPackage() != null && printQualified(ref)) {
				if (!ref.getPackage().isUnnamedPackage()) {
					scan(ref.getPackage());
				//	result.getTokenSequence().add(Pair.of(CtPackage.PACKAGE_SEPARATOR,null));
				}
			}

			if (ref.isParentInitialized() && !(ref.getParent() instanceof CtImport)) {
				this.writeAnnotations(ref);
			}
			result.getTokenSequence().add(Pair.of(ref.getSimpleName(),null));
		}
		
		if (withGenerics && !ignoreGenerics) {
			this.writeActualTypeArguments(ref);
		}
	}
	
	public void writeActualTypeArguments(CtActualTypeContainer ctGenericElementReference) {
		final Collection<CtTypeReference<?>> arguments = ctGenericElementReference.getActualTypeArguments();
		if (arguments != null && !arguments.isEmpty()) {
//			printList(arguments.stream().filter(a -> !a.isImplicit())::iterator,
//				null, "<", ",", ">",
//				argument -> {
//					if (this.forceWildcardGenerics) {
//						result.getTokenSequence().add(Pair.of("?",null));
//					} else {
//						this.scan(argument);
//					}
//				});
			printList(arguments.stream().filter(a -> !a.isImplicit())::iterator,
					null, "", "", "",
					argument -> {
						if (this.forceWildcardGenerics) {
							result.getTokenSequence().add(Pair.of("?",null));
						} else {
							this.scan(argument);
						}
					});
		 }
	}
	
	public <T> void printList(Iterable<T> iterable, String startKeyword, String start, String next, String end,
			Consumer<T> elementPrinter) {		
		if (startKeyword != null) {
			result.getTokenSequence().add(Pair.of(startKeyword,null));
		}
		
		try {
			if (start != null && !start.isEmpty()) {
				result.getTokenSequence().add(Pair.of(start,null));
			}	
			for (T item : iterable) {
				this.printSeparatorIfAppropriate(next);
				elementPrinter.accept(item);
			}
			isFirst = true;
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
		
		if (end != null && !end.isEmpty()) {
			result.getTokenSequence().add(Pair.of(end,null));
		}
	}
	
	public void printSeparatorIfAppropriate(String separator) {
		if (isFirst) {
			isFirst = false;
		} else {	
			if (separator != null && !separator.isEmpty()) {
				result.getTokenSequence().add(Pair.of(separator, null));
			}	
		}
	}
	
	private boolean printQualified(CtTypeReference<?> ref) {
		return ignoreImplicit || !ref.isSimplyQualified();
	}
	
	protected String stripLeadingDigits(String simpleName) {
		int i = 0;
		while (i < simpleName.length()) {
			if (!Character.isDigit(simpleName.charAt(i))) {
				return simpleName.substring(i);
			}
			i++;
		}
		return simpleName;
	}
	
	private boolean isPrintTypeReference(CtTypeReference<?> accessType) {	
		if (!accessType.isImplicit()) {
			return true;
		}
		
		if (this.forceWildcardGenerics && accessType.getTypeDeclaration().getFormalCtTypeParameters().size() > 0) {
			//print access type if access type is generic and we have to force wildcard generics
			/*
			 * E.g.
			 * class A<T> {
			 *  class B {
			 *  }
			 *  boolean m(Object o) {
			 *   return o instanceof B;			//compilation error
			 *   return o instanceof A.B; 		// OK
			 *   return o instanceof A<?>.B; 	// OK
			 *  }
			 * }
			 */
			return true;
		}
		return false;
	}
	
	public void scan(CtElement e) {
		if (e != null) {
			e.accept(this);
		}
	}
	
	@Override
	public <A extends Annotation> void visitCtAnnotationType(CtAnnotationType<A> annotationType) {
		visitCtType(annotationType);	
		result.getTokenSequence().add(Pair.of("@"+"interface",null));
		result.getTokenSequence().add(Pair.of(annotationType.getSimpleName(),null));
	//	result.getTokenSequence().add(Pair.of("{",null));
		this.writeElementList(annotationType.getTypeMembers());
	//	result.getTokenSequence().add(Pair.of("}",null));
	}
	
	@Override
	public void visitCtAnonymousExecutable(CtAnonymousExecutable impl) {
		this.writeAnnotations(impl);
		this.writeModifiers(impl);
		scan(impl.getBody());
	}
	
	@Override
	public <T> void visitCtArrayRead(CtArrayRead<T> arrayRead) {
		printCtArrayAccess(arrayRead);
	}
	
	@Override
	public <T> void visitCtArrayWrite(CtArrayWrite<T> arrayWrite) {
		printCtArrayAccess(arrayWrite);
	}
		
	private <T, E extends CtExpression<?>> void printCtArrayAccess(CtArrayAccess<T, E> arrayAccess) {
		enterCtExpression(arrayAccess);
		if (arrayAccess.getTarget() instanceof CtNewArray
				&& ((CtNewArray<?>) arrayAccess.getTarget()).getElements().isEmpty()) {
		//	result.getTokenSequence().add(Pair.of("(",null));
			scan(arrayAccess.getTarget());
		//	result.getTokenSequence().add(Pair.of(")",null));
		} else {
			scan(arrayAccess.getTarget());
		}
	//	result.getTokenSequence().add(Pair.of("[",null));
		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		scan(arrayAccess.getIndexExpression());
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        arrayIndexVarsAssigned = getInvolvedVariables(copybefore, copyafter);
	//	result.getTokenSequence().add(Pair.of("]",null));
		exitCtExpression(arrayAccess);
	}
	
	@Override
	public <T> void visitCtArrayTypeReference(CtArrayTypeReference<T> reference) {
		if (reference.isImplicit()) {
			return;
		}
		scan(reference.getComponentType());
		if (!SKIP_ARRAY) {
		//	result.getTokenSequence().add(Pair.of("[",null));
		//	result.getTokenSequence().add(Pair.of("]",null));
		}
	}
	
	@Override
	public <T> void visitCtAssert(CtAssert<T> asserted) {
		DependencyFlowNode nodeFromLastStatement=this.getLastNode();
		enterCtStatement(asserted);
		result.getTokenSequence().add(Pair.of("assert",null));
		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		scan(asserted.getAssertExpression());
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
        dealVariablesInGraph(involvedVariables,"others", null, nodeFromLastStatement, false);
        if (asserted.getExpression() != null) {
			result.getTokenSequence().add(Pair.of(":",null));
			scan(asserted.getExpression());
		}
		exitCtStatement(asserted);	
	}
	
	@Override
	public <T, A extends T> void visitCtAssignment(CtAssignment<T, A> assignement) {
		DependencyFlowNode nodeFromLastStatement=this.getLastNode();
		registerStatementLabel(assignement);
		enterCtStatement(assignement);
		enterCtExpression(assignement);

		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		scan(assignement.getAssigned());
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariablesAssigned = getInvolvedVariables(copybefore, copyafter);
        if(assignement.getAssigned() instanceof CtArrayAccess)
        	involvedVariablesAssigned.removeAll(arrayIndexVarsAssigned);
        dealVariablesInGraph(involvedVariablesAssigned,"Assigned", null, nodeFromLastStatement, false);

		result.getTokenSequence().add(Pair.of("=",null));
		ArrayList<Pair<String, CtElement>> copybeforeAssignment = new ArrayList<>(result.getTokenSequence());
		scan(assignement.getAssignment());
		ArrayList<Pair<String, CtElement>> copyafterAssignment = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariablesAssignement = getInvolvedVariables(copybeforeAssignment, copyafterAssignment); 

    	if(assignement.getAssignment() instanceof CtConditional) {
    		variablesConditionalThen.addAll(variablesConditionalElse);
    		involvedVariablesAssignement = new ArrayList<>(variablesConditionalThen);
    	} else if(assignement.getAssignment() instanceof CtSwitchExpression) {
    		involvedVariablesAssignement.clear();
		}  else if(assignement.getAssignment() instanceof CtLambda) {
			involvedVariablesAssignement.clear();
		} else if(assignement.getAssignment() instanceof CtNewClass) {
			involvedVariablesAssignement.clear();
		} 
        dealVariablesInGraph (involvedVariablesAssignement,"Assignment", null, nodeFromLastStatement, true);
        
		exitCtExpression(assignement);
		exitCtStatement(assignement);
	}
	
	@Override
	public <T> void visitCtBinaryOperator(CtBinaryOperator<T> operator) {
		enterCtExpression(operator);
		scan(operator.getLeftHandOperand());
		result.getTokenSequence().add(Pair.of(ExtendedCDGBuilder.getOperatorText(operator.getKind()),null));
		try {
			scan(operator.getRightHandOperand());
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
		exitCtExpression(operator);
	}
	
	@Override
	public <R> void visitCtBlock(CtBlock<R> block) {
		enterCtStatement(block);
		if (!block.isImplicit()) {
			// result.getTokenSequence().add(Pair.of("{",null));
		}
		for (CtStatement statement : block.getStatements()) {
			if (!statement.isImplicit()) {
				this.writeStatement(statement);
			}
		}
		if (!block.isImplicit()) {
			// result.getTokenSequence().add(Pair.of("}",null));
		}
		exitCtStatement(block);
	}
	
	public void writeStatement(CtStatement statement) {
		registerStatementLabel(statement);
		try {
			this.scan(statement);
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
	}
	
	@Override
	public void visitCtBreak(CtBreak breakStatement) {
		enterCtStatement(breakStatement);
		int breakindex=0;
		if (!breakStatement.isImplicit()) {
			result.getTokenSequence().add(Pair.of("break",null));
			breakindex = result.getTokenSequence().size();
			if (breakStatement.getTargetLabel() != null) {
				result.getTokenSequence().add(Pair.of(breakStatement.getTargetLabel(),null));
			}
		
			DependencyFlowNode lastNodeToUse = this.getLastNodeToUse();
			// currently assume we do not have control flow transfer because of break label
			DependencyFlowNode breakNode = new DependencyFlowNode(breakStatement, result, NodeKind.BRANCH, NodeType.Branchbreak,
					NodeRole.BLOCKDEFAULT, null, breakindex, ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
    		result.addVertex(breakNode);

			tryAddEdge(lastNodeToUse, breakNode);
			
			if (!breakingBad.empty()) {
				tryAddEdge(breakNode, breakingBad.peek(), false, true);
			}
		}
		exitCtStatement(breakStatement);
	}
	
	@Override
	@SuppressWarnings("rawtypes")
	public <E> void visitCtCase(CtCase<E> caseStatement) {
		registerStatementLabel(caseStatement);
		enterCtStatement(caseStatement);
		int caseindex=result.getTokenSequence().size()+1;
		DependencyFlowNode caseNode = new DependencyFlowNode(caseStatement, result, 
				NodeKind.BRANCH, NodeType.CaseExpression, NodeRole.BLOCKDEFAULT,null,caseindex, ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(caseNode);
		tryAddEdge(lastControlNode.peek(), caseNode);
		if (caseStatement.getCaseExpression() != null) {
			result.getTokenSequence().add(Pair.of("case",null));
			List<CtExpression<E>> caseExpressions = caseStatement.getCaseExpressions();
			for (int i = 0; i < caseExpressions.size(); i++) {
				CtExpression<E> caseExpression = caseExpressions.get(i);
				// writing enum case expression
				ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
				if (caseExpression instanceof CtFieldAccess) {
					final CtFieldReference variable = ((CtFieldAccess) caseExpression).getVariable();
					// In noclasspath mode, we don't have always the type of the declaring type.
					if (variable.getType() != null
							&& variable.getDeclaringType() != null
							&& variable.getType().getQualifiedName().equals(variable.getDeclaringType().getQualifiedName())) {
						result.getTokenSequence().add(Pair.of(variable.getSimpleName(), caseExpression));
					} else {
						scan(caseExpression);
					}
				} else {
					scan(caseExpression);
				}
				ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
		        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
		        dealVariablesInGraph(involvedVariables,"others", null, caseNode, false);
		        
				if (i != caseExpressions.size() - 1) {
				//	result.getTokenSequence().add(Pair.of(",",null));
				}
			}
		} else {
			result.getTokenSequence().add(Pair.of("default",null));
		}
		
		String separator = caseStatement.getCaseKind() == CaseKind.ARROW ? "->" : ":";
		result.getTokenSequence().add(Pair.of(separator,null));
		for (CtStatement statement : caseStatement.getStatements()) {
			this.writeStatement(statement);
		}
		exitCtStatement(caseStatement);
	}
	
	@Override
	public void visitCtCatch(CtCatch catchBlock) { }
	
	@Override
	public <T> void visitCtClass(CtClass<T> ctClass) {
		this.pushCurrentThis(ctClass);
		if (ctClass.getSimpleName() != null && !CtType.NAME_UNKNOWN.equals(ctClass.getSimpleName()) && !ctClass.isAnonymous()) {
			visitCtType(ctClass);
			result.getTokenSequence().add(Pair.of("class",null));
			result.getTokenSequence().add(Pair.of(stripLeadingDigits(ctClass.getSimpleName()),null));
			this.writeFormalTypeParameters(ctClass);
			this.writeExtendsClause(ctClass);
			this.writeImplementsClause(ctClass);
		}
	//	result.getTokenSequence().add(Pair.of("{",null));
		this.writeElementList(ctClass.getTypeMembers());
	//	result.getTokenSequence().add(Pair.of("}",null));
		this.popCurrentThis();
	}
	
	@Override
	public void visitCtTypeParameter(CtTypeParameter typeParameter) {
		this.writeAnnotations(typeParameter);
		result.getTokenSequence().add(Pair.of(typeParameter.getSimpleName(),null));
		if (typeParameter.getSuperclass() != null && typeParameter.getSuperclass().isImplicit() == false) {
			result.getTokenSequence().add(Pair.of("extends",null));
			scan(typeParameter.getSuperclass());
		}
	}
	
	public boolean parentassignment(CtElement element) {
		if(element!=null && element.getParent() instanceof CtAssignment)
			return true;
		else if(element!=null && element.getParent()!=null && !(element.getParent() instanceof CtBlock)
				&& element.getParent().getParent() instanceof CtAssignment)
			return true;
		else if(element!=null && element.getParent()!=null && !(element.getParent() instanceof CtBlock)
				&& element.getParent().getParent()!=null && element.getParent().getParent().getParent() instanceof CtAssignment)
			return true;
		else return false;
	}

	@Override
	public <T> void visitCtConditional(CtConditional<T> conditional) {
		DependencyFlowNode nodeFromLastStatement=this.getLastNodeToUse();
		enterCtExpression(conditional);
		outlevel++;
		checkisAssignmenrExper(conditional);
		CtExpression<Boolean> condition = conditional.getCondition();
		boolean parent;
		try {
			parent = conditional.getParent() instanceof CtAssignment || conditional.getParent() instanceof CtVariable;
		} catch (ParentNotInitializedException ex) {
			parent = false;
		}
		if (parent) {
		//	result.getTokenSequence().add(Pair.of("(",null));
		}
		
		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		scan(condition);
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
		if (parent) {
		//	result.getTokenSequence().add(Pair.of(")",null));
		}
		result.getTokenSequence().add(Pair.of("?",null)); 
		DependencyFlowNode ConditionalNode = new DependencyFlowNode (condition, result, NodeKind.BRANCH, 
				NodeType.Conditionalcondition, NodeRole.BLOCKDEFAULT, null, result.getTokenSequence().size(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(ConditionalNode);
		dealVariablesInGraph(involvedVariables,"others", ConditionalNode, nodeFromLastStatement, parentassignment(conditional));
		if(involvedVariables.size()==0)
		    tryAddEdge(nodeFromLastStatement, ConditionalNode);
		
		CtExpression<T> thenExpression = conditional.getThenExpression();
		ArrayList<Pair<String, CtElement>> copybeforethen = new ArrayList<>(result.getTokenSequence());
		innerindex++;
		scan(thenExpression);
		ArrayList<Pair<String, CtElement>> copyafterthen = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariablesthen = getInvolvedVariables(copybeforethen, copyafterthen);
		result.getTokenSequence().add(Pair.of(":",null));
		DependencyFlowNode ConditionalThenNode = new DependencyFlowNode (condition, result, NodeKind.BRANCH, 
				NodeType.ConditionalThen, NodeRole.BLOCKDEFAULT, null, result.getTokenSequence().size(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(ConditionalThenNode);
		dealVariablesInGraph(involvedVariablesthen,"others", null, ConditionalThenNode, parentassignment(conditional));
        tryAddEdge(ConditionalNode, ConditionalThenNode);
        variablesConditionalThen = involvedVariablesthen;
        DependencyFlowNode lasethenrelatednode=this.getLastNode();

		CtExpression<T> elseExpression = conditional.getElseExpression();
		boolean isAssign;
		if ((isAssign = elseExpression instanceof CtAssignment)) {
		//	result.getTokenSequence().add(Pair.of("(",null));
		}
		ArrayList<Pair<String, CtElement>> copybeforeelse = new ArrayList<>(result.getTokenSequence());
		innerindex++;
		scan(elseExpression);
		ArrayList<Pair<String, CtElement>> copyafterelse = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariableselse = getInvolvedVariables(copybeforeelse, copyafterelse);
		if (isAssign) {
		//	result.getTokenSequence().add(Pair.of(")",null));
		}
		DependencyFlowNode ConditionalElseNode = new DependencyFlowNode (condition, result, NodeKind.BRANCH, 
				NodeType.ConditionalElse, NodeRole.BLOCKDEFAULT, null, ConditionalThenNode.getTokenIndex(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(ConditionalElseNode);
		dealVariablesInGraph(involvedVariableselse,"others", null, ConditionalElseNode, parentassignment(conditional));
        tryAddEdge(ConditionalNode, ConditionalElseNode);
        variablesConditionalElse = involvedVariableselse;
        DependencyFlowNode laseelserelatednode=this.getLastNode();
        
        DependencyFlowNode convergenceNode = new DependencyFlowNode(null, result, 
				NodeKind.BRANCH, NodeType.ConditionalCONVERGE, NodeRole.BLOCKDEFAULT, null, ConditionalThenNode.getTokenIndex(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(convergenceNode);
        tryAddEdge(lasethenrelatednode, convergenceNode);
        tryAddEdge(laseelserelatednode, convergenceNode);
        outlevel--;
	    isExperAssignmentVars = false;
		exitCtExpression(conditional);
	}
	
	protected void registerStatementLabel(CtStatement st) {
		if (st.getLabel() == null || st.getLabel().isEmpty()) {
			return;
		}
		
		if (!labeledStatement.containsKey(st.getLabel())) {
			labeledStatement.put(st.getLabel(), st);
		}
	}
	
	@Override
	public <T> void visitCtConstructor(CtConstructor<T> constructor) {
		DependencyFlowNode nodeFromLastStatement=this.getLastNode();
		this.visitCtNamedElement(constructor);
		this.writeModifiers(constructor);
		this.writeFormalTypeParameters(constructor);
		
		if (constructor.getDeclaringType() != null) {
			result.getTokenSequence().add(Pair.of(stripLeadingDigits(constructor.getDeclaringType().getSimpleName()),null));
		}
		if (!constructor.isCompactConstructor()) {
			ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
			this.writeExecutableParameters(constructor);
			ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
	        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
	        dealVariablesInGraph(involvedVariables,"others", null, nodeFromLastStatement, false);
		}
		this.writeThrowsClause(constructor);
		scan(constructor.getBody());
	}
	
	@Override
	public void visitCtContinue(CtContinue continueStatement) {
		enterCtStatement(continueStatement);
		int continueindex=0;
		result.getTokenSequence().add(Pair.of("continue",null));
		continueindex = result.getTokenSequence().size();
		if (continueStatement.getTargetLabel() != null) {
			result.getTokenSequence().add(Pair.of(continueStatement.getTargetLabel(),null));
		}
		
		DependencyFlowNode to = continueBad.peek();
		DependencyFlowNode lastNode = this.getLastNodeToUse();
		DependencyFlowNode continueNode = new DependencyFlowNode(continueStatement, result, NodeKind.BRANCH, NodeType.Branchcontinue,
				NodeRole.BLOCKDEFAULT, null, continueindex, ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(continueNode);
		tryAddEdge(lastNode, continueNode);
		if (to != null) {
			tryAddEdge(continueNode, to, true, false);
		}	
		exitCtStatement(continueStatement);
	}
	
	@Override
	public void visitCtDo(CtDo doLoop) {
		enterCtStatement(doLoop);
		registerStatementLabel(doLoop);	
		DependencyFlowNode nodeToUse=this.getLastNodeToUse();

		result.getTokenSequence().add(Pair.of("do",null));
		DependencyFlowNode convergenceNode = new DependencyFlowNode(null, result, NodeKind.BRANCH,
				NodeType.DoInsideCONVERGE, NodeRole.BLOCKDEFAULT, null, result.getTokenSequence().size(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(convergenceNode);
		continueBad.push(convergenceNode);
		
		DependencyFlowNode convergenceNodeout = new DependencyFlowNode(null, result, NodeKind.BRANCH,
				NodeType.DoOutsideCONVERGE, NodeRole.BLOCKDEFAULT, null, result.getTokenSequence().size(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(convergenceNodeout);
		breakingBad.push(convergenceNodeout);

		tryAddEdge(nodeToUse, convergenceNode);
		this.writeIfOrLoopBlock(doLoop.getBody());
		result.getTokenSequence().add(Pair.of("while",null));
		int whileindex = result.getTokenSequence().size();
		
		DependencyFlowNode nodeFromLastStatement=this.getLastNodeToUse();	
	//	result.getTokenSequence().add(Pair.of("(",null));	
		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		scan(doLoop.getLoopingExpression());
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
        DependencyFlowNode branch = new DependencyFlowNode(doLoop.getLoopingExpression(), result, NodeKind.BRANCH,
				NodeType.Dowhilecondition, NodeRole.BLOCKDEFAULT, null, whileindex, ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(branch);
		tryAddEdge(branch, convergenceNode, true, false);
		tryAddEdge(branch, convergenceNodeout);
        dealVariablesInGraph(involvedVariables,"others", branch, nodeFromLastStatement, false);
        if(involvedVariables.size()==0)
		    tryAddEdge(nodeFromLastStatement, branch);
	//	result.getTokenSequence().add(Pair.of(")",null));

		breakingBad.pop();
		continueBad.pop();
		lastConverengenceNode = convergenceNodeout;
		exitCtStatement(doLoop);
	}
	
	@Override
	public <T extends Enum<?>> void visitCtEnum(CtEnum<T> ctEnum) {
		visitCtType(ctEnum);
		result.getTokenSequence().add(Pair.of("enum",null));
		result.getTokenSequence().add(Pair.of(stripLeadingDigits(ctEnum.getSimpleName()),null));
		this.writeImplementsClause(ctEnum);
		this.pushCurrentThis(ctEnum);
	//	result.getTokenSequence().add(Pair.of("{",null));

		if (ctEnum.getEnumValues().isEmpty()) {
			result.getTokenSequence().add(Pair.of(";",null));
		} else {
//			this.printList(ctEnum.getEnumValues(),
//					null, null, ",", ";",
//					enumValue -> {
//						scan(enumValue);
//					});
			this.printList(ctEnum.getEnumValues(),
					null, null, "", ";",
					enumValue -> {
						scan(enumValue);
					});
		}

		this.writeElementList(ctEnum.getTypeMembers());
//		result.getTokenSequence().add(Pair.of("}",null));
		this.popCurrentThis();
	}
	
	@Override
	public <T> void visitCtExecutableReference(CtExecutableReference<T> reference) {
		result.getTokenSequence().add(Pair.of(reference.getSignature(),null));
	}
	
	@Override
	public <T> void visitCtField(CtField<T> f) {
		DependencyFlowNode nodeFromLastStatement=this.getLastNode();
		this.visitCtNamedElement(f);
		this.writeModifiers(f);
		scan(f.getType());
		ArrayList<Pair<String, CtElement>> copybeforeassigned = new ArrayList<>(result.getTokenSequence());
		result.getTokenSequence().add(Pair.of(f.getSimpleName(),f));
		ArrayList<Pair<String, CtElement>> copyafterassigned = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariablesassigned = getInvolvedVariables(copybeforeassigned, copyafterassigned);
        if(f.getType() instanceof CtArrayTypeReference)
        	involvedVariablesassigned.removeAll(arrayIndexVarsAssigned);
        dealVariablesInGraph(involvedVariablesassigned,"Assigned", null, nodeFromLastStatement, false);
                
		ArrayList<Pair<String, CtElement>> copybeforeassignment = new ArrayList<>(result.getTokenSequence());
		if (f.getDefaultExpression() != null) {
			result.getTokenSequence().add(Pair.of("=",null));
			scan(f.getDefaultExpression());
		}
		result.getTokenSequence().add(Pair.of(";",null));
		ArrayList<Pair<String, CtElement>> copyafterassignment = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariablesassignment = getInvolvedVariables(copybeforeassignment, copyafterassignment);
    	
    	if(f.getDefaultExpression() instanceof CtConditional) {
    		variablesConditionalThen.addAll(variablesConditionalElse);
    		involvedVariablesassignment = new ArrayList<>(variablesConditionalThen);
    	} else if(f.getDefaultExpression() instanceof CtSwitchExpression) {
    		involvedVariablesassignment.clear();
		}  else if(f.getDefaultExpression() instanceof CtLambda) {
			involvedVariablesassignment.clear();
		} else if(f.getDefaultExpression() instanceof CtNewClass) {
			involvedVariablesassignment.clear();
		} 
        dealVariablesInGraph(involvedVariablesassignment,"Assignment", null, nodeFromLastStatement, true);
	}
	
	@Override
	public <T> void visitCtEnumValue(CtEnumValue<T> enumValue) {
		this.visitCtNamedElement(enumValue);
		result.getTokenSequence().add(Pair.of(enumValue.getSimpleName(),null));
		if (enumValue.getDefaultExpression() != null) {
			CtConstructorCall<?> constructorCall = (CtConstructorCall<?>) enumValue.getDefaultExpression();
			if (!constructorCall.isImplicit()) {
//				this.printList(constructorCall.getArguments(), null, "(", ",", ")", expr -> scan(expr));
				this.printList(constructorCall.getArguments(), null, "", "", "", expr -> scan(expr));
			}
			if (constructorCall instanceof CtNewClass) {
				scan(((CtNewClass<?>) constructorCall).getAnonymousClass());
			}
		}
	}
	
	@Override
	public <T> void visitCtFieldRead(CtFieldRead<T> fieldRead) {
		printCtFieldAccess(fieldRead);
	}
	
	@Override
	public <T> void visitCtFieldWrite(CtFieldWrite<T> fieldWrite) {
		printCtFieldAccess(fieldWrite);
	}
	
	private <T> void printCtFieldAccess(CtFieldAccess<T> f) {
		enterCtExpression(f);
		try  {
			if ((f.getVariable().isStatic() || "class".equals(f.getVariable().getSimpleName()))
					&& f.getTarget() instanceof CtTypeAccess) {
				this.ignoreGenerics=true;
			}
			CtExpression<?> target = f.getTarget();
			if (target != null) {
					// the implicit drives the separator
				if (shouldPrintTarget(target)) {
						scan(target);
					//	result.getTokenSequence().add(Pair.of(".",null));
					}
				this.IGNORE_STATIC_ACCESS=true;
			}
			scan(f.getVariable());
			this.ignoreGenerics=false;
			this.IGNORE_STATIC_ACCESS=false;
		} catch(Exception e) {
			System.out.println(e.getMessage());
		}
		exitCtExpression(f);
	}
	
	@Override
	public <T> void visitCtThisAccess(CtThisAccess<T> thisAccess) {
		try {
			enterCtExpression(thisAccess);
			// we only write qualified this when this is required
			// this is good both in fully-qualified mode and in readable (with-imports) mode
			// the implicit information is used for analysis (e.g. is visibility caused by implicit bugs?) but
			// not for pretty-printing
			CtTypeAccess<?> target = (CtTypeAccess<?>) thisAccess.getTarget();
			CtTypeReference<?> targetType = target.getAccessedType();

			// readable mode as close as possible to the original code
			if (thisAccess.isImplicit()) {
				// write nothing, "this" is implicit and we unfortunately cannot always know
				// what the good target is in JDTTreeBuilder
				return;
			}

			// the simplest case: we always print "this" if we're in the top-level class,
			// this is shorter (no qualified this), explicit, and less fragile wrt transformation
			if (targetType == null || (thisAccess.getParent(CtType.class) != null && thisAccess.getParent(CtType.class).isTopLevel())) {
				result.getTokenSequence().add(Pair.of("this",null));
				return; // still go through finally block below
			}

			// we cannot have fully-qualified this in anonymous classes
			// we simply print "this" and it always works
			// this has to come after the implicit test just before
			if (targetType.isAnonymous()) {
				result.getTokenSequence().add(Pair.of("this",null));
				return;
			}

			// complex case of qualified this
			if (!this.currentThis.isEmpty()) {

				CtType<?> lastType = this.currentThis.peekFirst();
				String lastTypeQualifiedName = lastType.getQualifiedName();
				String targetTypeQualifiedName = targetType.getQualifiedName();

				if (!lastTypeQualifiedName.equals(targetTypeQualifiedName)) {
					if (!targetType.isImplicit()) {
						visitCtTypeReferenceWithoutGenerics(targetType);
					//	result.getTokenSequence().add(Pair.of(".",null));
					}
					result.getTokenSequence().add(Pair.of("this",null));
					return;
				}
			}
			result.getTokenSequence().add(Pair.of("this",null));
		} finally {
			exitCtExpression(thisAccess);
		}
	}
	
	@Override
	public <T> void visitCtSuperAccess(CtSuperAccess<T> f) {
		enterCtExpression(f);
		if (f.getTarget() != null) {
			scan(f.getTarget());
		//	result.getTokenSequence().add(Pair.of(".",null));
		}
		result.getTokenSequence().add(Pair.of("super",null));
		exitCtExpression(f);
	}
	
	@Override
	public void visitCtJavaDoc(CtJavaDoc comment) { }
	
	@Override
	public void visitCtComment(CtComment comment) { }
	
	@Override
	public void visitCtJavaDocTag(CtJavaDocTag docTag) { }
	
	@Override
	public void visitCtImport(CtImport ctImport) { }
	
	@Override
	public void visitCtModule(CtModule module) { }
	
	@Override
	public void visitCtModuleReference(CtModuleReference moduleReference) { }
	
	@Override
	public void visitCtPackageExport(CtPackageExport moduleExport) { }
	
	@Override
	public void visitCtPackageReference(CtPackageReference reference) { }
	
	@Override
	public void visitCtModuleRequirement(CtModuleRequirement moduleRequirement) { }
	
	@Override
	public void visitCtProvidedService(CtProvidedService moduleProvidedService) { }
	
	@Override
	public void visitCtUsedService(CtUsedService usedService) { }
	
	@Override
	public void visitCtCompilationUnit(CtCompilationUnit compilationUnit) { }
	
	@Override
	public void visitCtPackageDeclaration(CtPackageDeclaration packageDeclaration) { }
	
	@Override
	public void visitCtTypeMemberWildcardImportReference(CtTypeMemberWildcardImportReference wildcardReference) {
		scan(wildcardReference.getTypeReference());
	//	result.getTokenSequence().add(Pair.of(".*",null));
	}
	
	@Override
	public <T> void visitCtAnnotationFieldAccess(CtAnnotationFieldAccess<T> annotationFieldAccess) {
		enterCtExpression(annotationFieldAccess);
		try  {
			if (annotationFieldAccess.getTarget() != null) {
				scan(annotationFieldAccess.getTarget());
			//	result.getTokenSequence().add(Pair.of(".",null));
				this.IGNORE_STATIC_ACCESS=true;
			}
			this.ignoreGenerics=true;
			scan(annotationFieldAccess.getVariable());
		//	result.getTokenSequence().add(Pair.of("()",null));
			this.ignoreGenerics=false;
			this.IGNORE_STATIC_ACCESS=false;
		} catch(Exception e) {
			System.out.println(e.getMessage());
		}
		exitCtExpression(annotationFieldAccess);
	}
	
	@Override
	public <T> void visitCtFieldReference(CtFieldReference<T> reference) {
		boolean isStatic = "class".equals(reference.getSimpleName()) || !"super".equals(reference.getSimpleName()) && reference.isStatic();
		boolean printType = true;

		if (reference.isFinal() && reference.isStatic()) {
			CtTypeReference<?> declTypeRef = reference.getDeclaringType();
			if (declTypeRef.isAnonymous()) {
				//never print anonymous class ref
				printType = false;
			} else {
				if (this.isInCurrentScope(declTypeRef)) {
					//do not printType if we are in scope of that type
					printType = false;
				}
			}
		}

		if (isStatic && printType && !this.IGNORE_STATIC_ACCESS) {
			try  {
				this.IGNORE_STATIC_ACCESS=true;
				scan(reference.getDeclaringType());
				this.IGNORE_STATIC_ACCESS=false;
			} catch (Exception e) {
				System.out.println(e.getMessage());
			}
		//	result.getTokenSequence().add(Pair.of(".",null));
		}
		if ("class".equals(reference.getSimpleName())) {
			result.getTokenSequence().add(Pair.of("class",null));
		} else {
			result.getTokenSequence().add(Pair.of(reference.getSimpleName(),reference));
		}
	}
	
	public boolean isInCurrentScope(CtTypeReference<?> typeRef) {
		CtTypeReference<?> currentTypeRef = getCurrentTypeReference();
		return typeRef.equals(currentTypeRef);
	}
	
	public CtTypeReference<?> getCurrentTypeReference() {
		if (currentTopLevel != null) {
			CtType<?> tc = getCurrentTypeContext();
			if (tc != null) {
				return tc.getReference();
			}
			return currentTopLevel.getReference();
		}
		return null;
	}
	
	private CtType<?> getCurrentTypeContext() {
		if (currentThis != null && !currentThis.isEmpty()) {
			return currentThis.peek();
		}
		return null;
	}
	
	@Override
	public void visitCtFor(CtFor forLoop) {
		enterCtStatement(forLoop);
		registerStatementLabel(forLoop);
		result.getTokenSequence().add(Pair.of("for",null));
		int forindex=result.getTokenSequence().size();
	//	result.getTokenSequence().add(Pair.of("(",null));
		List<CtStatement> st = forLoop.getForInit();
		if (!st.isEmpty()) {
			try {
				this.FIRST_FOR_VARIABLE=true;
				scan(st.get(0));
				this.FIRST_FOR_VARIABLE=false;
			} catch(Exception e) {
				System.out.println(e.getMessage());
			}
		}
		if (st.size() > 1) {
			try {
				this.NEXT_FOR_VARIABLE = true;
				for (int i = 1; i < st.size(); i++) {
				//	result.getTokenSequence().add(Pair.of(",",null));
					scan(st.get(i));
				}
				this.NEXT_FOR_VARIABLE = false;
			} catch(Exception e) {
				System.out.println(e.getMessage());
			}
		}
		result.getTokenSequence().add(Pair.of(";",null));	

		DependencyFlowNode nodeFromLastStatement=this.getLastNodeToUse();
		DependencyFlowNode convergence = new DependencyFlowNode(null, result,
				NodeKind.BRANCH, NodeType.ForCONVERGE, NodeRole.BLOCKDEFAULT, null, result.getTokenSequence().size(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(convergence);
		breakingBad.push(convergence);
		
		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		scan(forLoop.getExpression());
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
		result.getTokenSequence().add(Pair.of(";",null));
		//Next the branch
		DependencyFlowNode branch = new DependencyFlowNode(forLoop.getExpression(), result, 
			NodeKind.BRANCH, NodeType.Forcondition, NodeRole.BLOCKDEFAULT, null, forindex, ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(branch);
		dealVariablesInGraph(involvedVariables,"others", branch, nodeFromLastStatement, false);
		if(involvedVariables.size()==0)
		    tryAddEdge(nodeFromLastStatement, branch);
		//Node continue statements can continue to
		continueBad.push(branch);
		
		this.writeIfOrLoopBlock(forLoop.getBody());
		
//		this.printList(forLoop.getForUpdate(),
//			null, null, ",", null, s -> scan(s));
		this.printList(forLoop.getForUpdate(),
				null, null, "", null, s -> scan(s));
	//	result.getTokenSequence().add(Pair.of(")",null));
		
		if (forLoop.getForUpdate() != null)
		    tryAddEdge(this.getLastNodeToUse(), branch, true, false);

		tryAddEdge(branch, convergence);
		continueBad.pop();
		breakingBad.pop();
		lastConverengenceNode = convergence;
		exitCtStatement(forLoop);
	}
	
	@Override
	public void visitCtForEach(CtForEach foreach) {
		enterCtStatement(foreach);
		registerStatementLabel(foreach);
		result.getTokenSequence().add(Pair.of("for",null));
	//	result.getTokenSequence().add(Pair.of("(",null));		
		scan(foreach.getVariable());
        
		result.getTokenSequence().add(Pair.of(":",null));
		DependencyFlowNode nodeFromLastStatement=this.getLastNodeToUse();
		DependencyFlowNode convergence = new DependencyFlowNode(null, result, NodeKind.BRANCH, 
				NodeType.ForEachCONVERGE, NodeRole.BLOCKDEFAULT, null, result.getTokenSequence().size(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(convergence);
		breakingBad.push(convergence);
		
		int conditionindex=result.getTokenSequence().size();
		ArrayList<Pair<String, CtElement>> copybeforeexper = new ArrayList<>(result.getTokenSequence());
		scan(foreach.getExpression());
		ArrayList<Pair<String, CtElement>> copyafterexper = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariablesexper = getInvolvedVariables(copybeforeexper, copyafterexper);
        DependencyFlowNode branch = new DependencyFlowNode(foreach.getExpression(), result, 
				NodeKind.BRANCH, NodeType.Foreachcondition, NodeRole.BLOCKDEFAULT, null, conditionindex, ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(branch);
        dealVariablesInGraph(involvedVariablesexper,"others", branch, nodeFromLastStatement, false);
        if(involvedVariablesexper.size()==0)
		    tryAddEdge(nodeFromLastStatement, branch);
		continueBad.push(branch);	
	//	result.getTokenSequence().add(Pair.of(")",null));
		
		int idbefore=this.getLastNode().getId();
		this.writeIfOrLoopBlock(foreach.getBody());
		int idafter=this.getLastNode().getId();
		if(idafter>idbefore)	
		    tryAddEdge(this.getLastNodeToUse(), branch, true, false);	
		tryAddEdge(branch, convergence);

		breakingBad.pop();
		continueBad.pop();
		lastConverengenceNode = convergence;
		exitCtStatement(foreach);
	}
	
	@Override
	public void visitCtIf(CtIf ifElement) {
		DependencyFlowNode nodeFromLastStatement=this.getLastNodeToUse();
		enterCtStatement(ifElement);
		outlevel++;
		registerStatementLabel(ifElement);
		result.getTokenSequence().add(Pair.of("if",null));
		int ifindex = result.getTokenSequence().size();
	//	result.getTokenSequence().add(Pair.of("(",null));
		
		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		scan(ifElement.getCondition());
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
        DependencyFlowNode branch = new DependencyFlowNode(ifElement.getCondition(), result, NodeKind.BRANCH,
				NodeType.Ifcondition, NodeRole.BLOCKDEFAULT, null, ifindex, ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(branch);
        dealVariablesInGraph(involvedVariables,"others", branch, nodeFromLastStatement, false);
        if(involvedVariables.size()==0)
		    tryAddEdge(nodeFromLastStatement, branch);
        
	//	result.getTokenSequence().add(Pair.of(")",null));
		DependencyFlowNode convergenceNode = new DependencyFlowNode(null, result, NodeKind.BRANCH, 
				NodeType.IfCONVERGE, NodeRole.BLOCKDEFAULT, null, result.getTokenSequence().size(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(convergenceNode);

		CtStatement thenStmt = ifElement.getThenStatement();
		CtStatement elseStmt = ifElement.getElseStatement();
		if(thenStmt!=null) {
			DependencyFlowNode ifThenNode = new DependencyFlowNode (null, result, NodeKind.BRANCH, 
					NodeType.IfThen, NodeRole.BLOCKDEFAULT, null, result.getTokenSequence().size(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
			result.addVertex(ifThenNode);
			tryAddEdge(branch, ifThenNode);
			innerindex++;
		    this.writeIfOrLoopBlock(thenStmt);		    
			tryAddEdge(this.getLastNodeToUse(), convergenceNode);
		}
		if (elseStmt != null) {
			result.getTokenSequence().add(Pair.of("else",null));
			DependencyFlowNode ifElseNode = new DependencyFlowNode (null, result, NodeKind.BRANCH, 
					NodeType.IfElse, NodeRole.BLOCKDEFAULT, null, result.getTokenSequence().size(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
			result.addVertex(ifElseNode);
			tryAddEdge(branch, ifElseNode);
			innerindex++;
			if (this.isElseIf(ifElement)) {
				CtIf child;
				if (elseStmt instanceof CtBlock) {
					child = ((CtBlock<?>) elseStmt).getStatement(0);
				} else {
					child = (CtIf) elseStmt;
				}
				scan(child);
			} else {
				this.writeIfOrLoopBlock(elseStmt);
			}
			tryAddEdge(this.getLastNodeToUse(), convergenceNode);
		} else {
			tryAddEdge(branch, convergenceNode);
		}

		outlevel--;
		lastConverengenceNode = convergenceNode;
		exitCtStatement(ifElement);
	}
	
	public boolean isElseIf(CtIf ifStmt) {
		if (ifStmt.getElseStatement() == null) {
			return false;
		}
		if (ifStmt.getElseStatement() instanceof CtIf)  {
			return true;
		}
		if (ifStmt.getElseStatement() instanceof CtBlock) {
			CtBlock<?> block = (CtBlock<?>) ifStmt.getElseStatement();
			return ((block.getStatements().size() == 1) && (block.getStatement(0) instanceof CtIf));
		}
		return false;
	}
	
	@Override
	public <T> void visitCtInterface(CtInterface<T> intrface) {
		visitCtType(intrface);
		result.getTokenSequence().add(Pair.of("interface",null));
		result.getTokenSequence().add(Pair.of(stripLeadingDigits(intrface.getSimpleName()),null));
		if (intrface.getFormalCtTypeParameters() != null) {
			this.writeFormalTypeParameters(intrface);
		}

		if (!intrface.getSuperInterfaces().isEmpty()) {
			this.printList(intrface.getSuperInterfaces(),
				"extends", null, "", null, ref -> scan(ref));
//			this.printList(intrface.getSuperInterfaces(),
//				"extends", null, ",", null, ref -> scan(ref));
		}
		this.pushCurrentThis(intrface);
//		result.getTokenSequence().add(Pair.of("{",null));
		
		this.writeElementList(intrface.getTypeMembers());
//		result.getTokenSequence().add(Pair.of("}",null));
		this.popCurrentThis();
	}
	
	public void checkisAssignmenrExper (CtElement element) {
		if(element.getParent() instanceof CtAssignment || (element.getParent()!=null && element.getParent().getParent() instanceof CtAssignment)
				|| (element.getParent().getParent()!=null && element.getParent().getParent().getParent() instanceof CtAssignment))
			isExperAssignmentVars = true;
	}
	
	@Override
	public <T> void visitCtInvocation(CtInvocation<T> invocation) {
		DependencyFlowNode nodeFromLastStatement=this.getLastNode();
		enterCtStatement(invocation);
		registerStatementLabel(invocation);
		enterCtExpression(invocation);
		
		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		if (invocation.getExecutable().isConstructor()) {
			// It's a constructor (super or this)
			this.writeActualTypeArguments(invocation.getExecutable());
			CtType<?> parentType = invocation.getParent(CtType.class);
			if (parentType == null || parentType.getQualifiedName() != null 
					&& parentType.getQualifiedName().equals
					(invocation.getExecutable().getDeclaringType().getQualifiedName())) {
				result.getTokenSequence().add(Pair.of("this",null));
			} else {
				if (invocation.getTarget() != null && !invocation.getTarget().isImplicit()) {
					scan(invocation.getTarget());
				//	result.getTokenSequence().add(Pair.of(".",null));
				}
				result.getTokenSequence().add(Pair.of("super",null));
			}
		} else {
			// It's a method invocation
			if (invocation.getTarget() != null && (ignoreImplicit || 
					!invocation.getTarget().isImplicit())) {
				try {
					if (invocation.getTarget() instanceof CtTypeAccess) {
						this.ignoreGenerics=true;
					}
					if (shouldPrintTarget(invocation.getTarget())) {
						scan(invocation.getTarget());
					//	result.getTokenSequence().add(Pair.of(".",null));
					}
					this.ignoreGenerics=false;
				} catch(Exception e) {
					System.out.println(e.getMessage());
				}
			}
			this.writeActualTypeArguments(invocation);
			result.getTokenSequence().add(Pair.of(invocation.getExecutable().getSimpleName(),null));
		}
		
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>();
		if(!whetherContainLambada(invocation.getArguments()) && !whetherContainAnonymousCall(invocation.getArguments())) { //assume only one lambda if any
//		   this.printList(invocation.getArguments(),
//				null, "(", ",", ")", e -> scan(e));
		   this.printList(invocation.getArguments(),
					null, "", "", "", e -> scan(e));
		   copyafter = new ArrayList<>(result.getTokenSequence());
	    } else {
	       copyafter = new ArrayList<>(result.getTokenSequence());
//	       this.printList(invocation.getArguments(),
//					null, "(", ",", ")", e -> scan(e));
	       this.printList(invocation.getArguments(),
					null, "", "", "", e -> scan(e));
	    }
	    ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
	    checkisAssignmenrExper(invocation);
	    dealVariablesInGraph(involvedVariables,"others", null, nodeFromLastStatement, parentassignment(invocation));
	    isExperAssignmentVars = false;
		exitCtExpression(invocation);
		exitCtStatement(invocation);
	}
	
	public boolean whetherContainLambada(List<CtExpression<?>> arguments) {
		for(int index=0; index<arguments.size(); index++) {
			if(arguments.get(index) instanceof CtLambda)
				return true;
		}
		return false;
	}
	
	public boolean whetherContainAnonymousCall(List<CtExpression<?>> arguments) {
		for(int index=0; index<arguments.size(); index++) {
			if(arguments.get(index) instanceof CtNewClass)
				return true;
		}
		return false;
	}
	
	@Override
	public <T> void visitCtLiteral(CtLiteral<T> literal) {
		enterCtExpression(literal);
		result.getTokenSequence().add(Pair.of(literal.toString(),null));
		exitCtExpression(literal);
	}
	
	@Override
	public void visitCtTextBlock(CtTextBlock ctTextBlock) {
		enterCtExpression(ctTextBlock);
		result.getTokenSequence().add(Pair.of(ExtendedCDGBuilder.getTextBlockToken(ctTextBlock),null));
		exitCtExpression(ctTextBlock);
	}
	
	public static String getTextBlockToken(CtTextBlock literal) {
		String token = "\"\"\"\n"
				+ literal.getValue().replace("\\", "\\\\")
				+ "\"\"\"";
		return token;
	}
	
	@Override
	public <T> void visitCtLocalVariable(CtLocalVariable<T> localVariable) {
		DependencyFlowNode nodeFromLastStatement=this.getLastNode();
		registerStatementLabel(localVariable);
		enterCtStatement(localVariable);

		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		if (!this.NEXT_FOR_VARIABLE
				&& !localVariable.isImplicit() // for resources in try-with-resources
		) {
			this.writeModifiers(localVariable);
			if (localVariable.isInferred() ) {
				result.getTokenSequence().add(Pair.of("var",null));
			} else {
				if (localVariable.getType() instanceof CtArrayTypeReference<?>) {
					try  {
						scan(localVariable.getType());
					} catch (Exception e) {
						System.out.println(e.getMessage());
					}
				} else {
					scan(localVariable.getType());
				}
			}
		}
		result.getTokenSequence().add(Pair.of(localVariable.getSimpleName(),localVariable));
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
        if(localVariable.getType() instanceof CtArrayTypeReference)
        	involvedVariables.removeAll(arrayIndexVarsAssigned);
        dealVariablesInGraph(involvedVariables,"Assigned", null, nodeFromLastStatement, false);
        
		ArrayList<Pair<String, CtElement>> copybeforeassignment = new ArrayList<>(result.getTokenSequence());
		if (localVariable.getDefaultExpression() != null
				&& !localVariable.isImplicit() // for resources in try-with-resources
		) {
			result.getTokenSequence().add(Pair.of("=",null));
			scan(localVariable.getDefaultExpression());
		}
		ArrayList<Pair<String, CtElement>> copyafterassignment = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariablesassignment = getInvolvedVariables(copybeforeassignment, copyafterassignment);
    	
    	if(localVariable.getDefaultExpression() instanceof CtConditional) {
    		variablesConditionalThen.addAll(variablesConditionalElse);
    		involvedVariablesassignment = new ArrayList<>(variablesConditionalThen);
    	} else if(localVariable.getDefaultExpression() instanceof CtSwitchExpression) {
    		involvedVariablesassignment.clear();
		}  else if(localVariable.getDefaultExpression() instanceof CtLambda) {
			involvedVariablesassignment.clear();
		} else if(localVariable.getDefaultExpression() instanceof CtNewClass) {
			involvedVariablesassignment.clear();
		}	
        dealVariablesInGraph(involvedVariablesassignment,"Assignment", null, nodeFromLastStatement, true);
        exitCtStatement(localVariable);	
	}
	
	@Override
	public <T> void visitCtLocalVariableReference(CtLocalVariableReference<T> reference) {
		result.getTokenSequence().add(Pair.of(reference.getSimpleName(),reference));
	}
	
	@Override
	public <T> void visitCtCatchVariable(CtCatchVariable<T> catchVariable) {	
		this.writeModifiers(catchVariable);
		scan(catchVariable.getType());
		result.getTokenSequence().add(Pair.of(catchVariable.getSimpleName(),catchVariable));
	}
	
	@Override
	public <T> void visitCtCatchVariableReference(CtCatchVariableReference<T> reference) {
		result.getTokenSequence().add(Pair.of(reference.getSimpleName(),reference));
	}
	
	@Override
	public <T> void visitCtMethod(CtMethod<T> m) {
		this.visitCtNamedElement(m);
		this.writeModifiers(m);
		this.writeFormalTypeParameters(m);
		
		try  {
			this.ignoreGenerics=false;
			scan(m.getType());
		} catch(Exception e) {
			System.out.println(e.getMessage());
		}

		result.getTokenSequence().add(Pair.of(m.getSimpleName(),null));
		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		this.writeExecutableParameters(m);
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
        dealVariablesInGraph(involvedVariables,"others", null, this.getLastNode(), false);
		this.writeThrowsClause(m);
		if (m.getBody() != null) {
			scan(m.getBody());
		} else {
			result.getTokenSequence().add(Pair.of(";",null));
		}
	}
	
	@Override
	public <T> void visitCtAnnotationMethod(CtAnnotationMethod<T> annotationMethod) {
		this.visitCtNamedElement(annotationMethod);
		this.writeModifiers(annotationMethod);
		scan(annotationMethod.getType());
		result.getTokenSequence().add(Pair.of(annotationMethod.getSimpleName(),null));

	//	result.getTokenSequence().add(Pair.of("(",null));
	//	result.getTokenSequence().add(Pair.of(")",null));
		if (annotationMethod.getDefaultExpression() != null) {
			result.getTokenSequence().add(Pair.of("default",null));
			scan(annotationMethod.getDefaultExpression());
		}
		result.getTokenSequence().add(Pair.of(";",null));
	}
	
	@Override
	@SuppressWarnings("rawtypes")
	public <T> void visitCtNewArray(CtNewArray<T> newArray) {
		enterCtExpression(newArray);
		boolean isNotInAnnotation = (newArray.getParent(CtAnnotationType.class) == null)
				&& (newArray.getParent(CtAnnotation.class) == null);
		if (isNotInAnnotation) {
			CtTypeReference<?> ref = newArray.getType();

			if (ref != null) {
				result.getTokenSequence().add(Pair.of("new",null));
			}

			try {
				this.SKIP_ARRAY=true;
				scan(ref);
				this.SKIP_ARRAY=false;
			} catch (Exception e) {
				System.out.println(e.getMessage());
			}
			
			for (int i = 0; ref instanceof CtArrayTypeReference; i++) {
			//	result.getTokenSequence().add(Pair.of("[",null));
				if (newArray.getDimensionExpressions().size() > i) {
					CtExpression<Integer> e = newArray.getDimensionExpressions().get(i);
					scan(e);
				}
			//	result.getTokenSequence().add(Pair.of("]",null));
				ref = ((CtArrayTypeReference) ref).getComponentType();
			}
		}
		
		if (newArray.getDimensionExpressions().isEmpty()) {
//			this.printList(newArray.getElements(),
//				null, "{", ",", "}", e -> scan(e));
			this.printList(newArray.getElements(),
					null, "", "", "", e -> scan(e));
		}
		exitCtExpression(newArray);
	}
	
	@Override
	public <T> void visitCtConstructorCall(CtConstructorCall<T> ctConstructorCall) {
		DependencyFlowNode nodeFromLastStatement=this.getLastNode();
		enterCtStatement(ctConstructorCall);
		registerStatementLabel(ctConstructorCall);
		enterCtExpression(ctConstructorCall);
		
		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		try {
			if (ctConstructorCall.getTarget() != null) {
				scan(ctConstructorCall.getTarget());
			//	result.getTokenSequence().add(Pair.of(".",null));
				this.ignoreEnclosingClass=true;
			}
			if (hasDeclaringTypeWithGenerics(ctConstructorCall.getType())) {
				this.ignoreEnclosingClass=true;
			}
			result.getTokenSequence().add(Pair.of("new",null));
			if (!ctConstructorCall.getActualTypeArguments().isEmpty()) {
				this.writeActualTypeArguments(ctConstructorCall);
			}
			scan(ctConstructorCall.getType());
			this.ignoreEnclosingClass=false;	
		} catch(Exception e) {
			System.out.println(e.getMessage());
		}
		
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>();		
		if(!whetherContainLambada(ctConstructorCall.getArguments()) && !whetherContainAnonymousCall(ctConstructorCall.getArguments())) { //assume only one lambda if any
		//	this.printList(ctConstructorCall.getArguments(), null, "(", ",", ")", exp -> scan(exp));
			this.printList(ctConstructorCall.getArguments(), null, "", "", "", exp -> scan(exp));
			copyafter = new ArrayList<>(result.getTokenSequence());
		} else {
		    copyafter = new ArrayList<>(result.getTokenSequence());
		//    this.printList(ctConstructorCall.getArguments(), null, "(", ",", ")", exp -> scan(exp));
		    this.printList(ctConstructorCall.getArguments(), null, "", "", "", exp -> scan(exp));
		}
        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
        checkisAssignmenrExper(ctConstructorCall);
        dealVariablesInGraph (involvedVariables,"others", null, nodeFromLastStatement, parentassignment(ctConstructorCall));
	    isExperAssignmentVars = false;
        
		exitCtExpression(ctConstructorCall);
		exitCtStatement(ctConstructorCall);
	}
				        	   	
	@Override
	public <T> void visitCtNewClass(CtNewClass<T> newClass) {
		enterCtStatement(newClass);
		enterCtExpression(newClass);
		DependencyFlowNode nodebefore = this.getLastNodeToUse();
		int newclassindex = 0;
		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		try {
			if (newClass.getTarget() != null) {
				scan(newClass.getTarget());
			//	result.getTokenSequence().add(Pair.of(".",null));
				this.ignoreEnclosingClass=true;
			}
			if (hasDeclaringTypeWithGenerics(newClass.getType())) {
				this.ignoreEnclosingClass=true;
			}
			result.getTokenSequence().add(Pair.of("new",null));
			newclassindex = result.getTokenSequence().size();
		
			if (!newClass.getActualTypeArguments().isEmpty()) {
				this.writeActualTypeArguments(newClass);
			}
			scan(newClass.getType());
			this.ignoreEnclosingClass=false;		
		} catch(Exception e) {
			System.out.println(e.getMessage());
		}
	//	this.printList(newClass.getArguments(), null, "(", ",", ")", exp -> scan(exp));
		this.printList(newClass.getArguments(), null, "", "", "", exp -> scan(exp));
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());		
        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
        dealVariablesInGraph (involvedVariables,"others", null, nodebefore, parentassignment(newClass));
		
        DependencyFlowNode NewClassEntry = new DependencyFlowNode(null, result, NodeKind.BRANCH, 
				NodeType.Newclassentry, NodeRole.BLOCKDEFAULT, null, newclassindex, ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(NewClassEntry);
		if(involvedVariables.size()>0)
		    tryAddEdge(this.getLastNodeToUse(), NewClassEntry);			
		else if(lastlastnodetouse!=null)
		    tryAddEdge(lastlastnodetouse, NewClassEntry);			
        
    	List<CtMethod<?>> containedMethods = newClass.getElements(new TypeFilter<>(CtMethod.class));
    	if(containedMethods.size()==1) { //we currently deal with situations where the anonymous class contains a single method	
    		scan(newClass.getAnonymousClass()); 
    	}
    	
    	DependencyFlowNode last= this.getLastNodeToUse();
		DependencyFlowNode NewClassExit = new DependencyFlowNode(null, result, NodeKind.BRANCH, 
				NodeType.Newclassexit, NodeRole.BLOCKDEFAULT, null, result.getTokenSequence().size(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(NewClassExit);
		tryAddEdge(last, NewClassExit);			
		if(newClass.getParent() instanceof CtAssignment || newClass.getParent() instanceof CtReturn)
			tryAddEdge(NewClassExit, nodebefore, true, false);
		
		exitCtExpression(newClass);
		exitCtStatement(newClass);
	}
	
	/**
	 * JDT doesn't support <code>new Foo<K>.Bar()</code>. To avoid reprinting this kind of type reference,
	 * we check that the reference has a declaring type with generics.
	 * See https://bugs.eclipse.org/bugs/show_bug.cgi?id=474593
	 *
	 * @param reference Type reference concerned by the bug.
	 * @return true if a declaring type has generic types.
	 */
	private <T> boolean hasDeclaringTypeWithGenerics(CtTypeReference<T> reference) {
		// We don't have a declaring type, it can't have generics.
		if (reference == null) {
			return false;
		}
		// If the declaring type isn't a type, we don't need this hack.
		if (reference.getDeclaringType() == null) {
			return false;
		}
		// If current reference is a class declared in a method, we don't need this hack.
		if (reference.isLocalType()) {
			return false;
		}
		// If declaring type have generics, we return true.
		if (!reference.getDeclaringType().getActualTypeArguments().isEmpty()) {
			return true;
		}
		// Checks if the declaring type has generic types.
		return hasDeclaringTypeWithGenerics(reference.getDeclaringType());
	}

	@Override
	public <T> void visitCtLambda(CtLambda<T> lambda) {
		enterCtExpression(lambda);
		DependencyFlowNode lastNodeToUse = this.getLastNodeToUse();
		DependencyFlowNode LambdaentryNode = new DependencyFlowNode(lambda, result, NodeKind.BRANCH, NodeType.Lambdaentry,
				NodeRole.BLOCKDEFAULT, null, result.getTokenSequence().size(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(LambdaentryNode);
		if(lastlastnodetouse!=null)
		    tryAddEdge(lastlastnodetouse, LambdaentryNode);
		
		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		// single parameter lambdas with implicit type can be printed without parantheses
		if (isSingleParameterWithoutExplicitType(lambda) && !ignoreImplicit) {
		//	this.printList(lambda.getParameters(), null, null, ",", null, this::scan);
			this.printList(lambda.getParameters(), null, null, "", null, this::scan);
		} else {
		//	this.printList(lambda.getParameters(), null, "(", ",", ")", this::scan);
			this.printList(lambda.getParameters(), null, "", "", "", this::scan);
		}
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
		result.getTokenSequence().add(Pair.of("->",null));
		LambdaentryNode.setTokenIndex(result.getTokenSequence().size());
		
		for(int index=0; index<involvedVariables.size();index++) {
        	Pair<CtElement, Integer> current=involvedVariables.get(index);
        	Pair<VariableType, String> varinfo = geVariableTypeAndShortname (current.getLeft());
        	CtElement element=current.getLeft();
        	Variable varstudy = new Variable (varinfo.getLeft(), VariableUseType.OTHERS,
        			 current.getRight(), varinfo.getRight(), element);	
       		DependencyFlowNode otherNode = new DependencyFlowNode (element, result,
       					NodeKind.VARIABLE, getNodeType(element), getNodeRole(element), varstudy, current.getRight(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);	
    		result.addVertex(otherNode);
       		tryAddEdge(LambdaentryNode, otherNode);	
        }

		if (lambda.getBody() != null) {
			scan(lambda.getBody());
		} else {
			ArrayList<Pair<String, CtElement>> copybeforeExper = new ArrayList<>(result.getTokenSequence());
			scan(lambda.getExpression());
			ArrayList<Pair<String, CtElement>> copyafterExper = new ArrayList<>(result.getTokenSequence());
	        ArrayList<Pair<CtElement, Integer>> involvedVariablesExper = getInvolvedVariables(copybeforeExper, copyafterExper);
	        dealVariablesInGraph (involvedVariablesExper,"others", null, null, parentassignment(lambda));
		}
		
		DependencyFlowNode lastNodeToUseOut = this.getLastNodeToUse();
		DependencyFlowNode LambdaexitNode = new DependencyFlowNode(lambda, result, NodeKind.BRANCH, NodeType.Lambdaexit,
				NodeRole.BLOCKDEFAULT, null, LambdaentryNode.getTokenIndex(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(LambdaexitNode);
		tryAddEdge(lastNodeToUseOut, LambdaexitNode);	
		if(lambda.getParent() instanceof CtAssignment && lambda.getRoleInParent() == CtRole.ASSIGNMENT)
			tryAddEdge(LambdaexitNode, lastNodeToUse, true, false);

		exitCtExpression(lambda);
	}
	
	private <T> boolean isSingleParameterWithoutExplicitType(CtLambda<T> lambda) {
		return lambda.getParameters().size() == 1 && (lambda.getParameters().get(0).getType() == null
				|| lambda.getParameters().get(0).getType().isImplicit());
	}
	
	@Override
	public <T, E extends CtExpression<?>> void visitCtExecutableReferenceExpression(
			CtExecutableReferenceExpression<T, E> expression) {
		enterCtExpression(expression);
		try  {
			if (expression.getExecutable().isStatic()) {
				this.ignoreGenerics=true;
			}
			scan(expression.getTarget());
			this.ignoreGenerics=false;
		} catch(Exception e) {
			System.out.println(e.getMessage());
		}
		result.getTokenSequence().add(Pair.of("::",null));
		
		if (!expression.getExecutable().getActualTypeArguments().isEmpty()) {
//			this.printList(expression.getExecutable().getActualTypeArguments(), 
//					null,"<", ", ", ">", this::scan);
			this.printList(expression.getExecutable().getActualTypeArguments(), 
					null,"", "", "", this::scan);
		}
		if (expression.getExecutable().isConstructor()) {
			result.getTokenSequence().add(Pair.of("new",null));
		} else {
			result.getTokenSequence().add(Pair.of(expression.getExecutable().getSimpleName(),null));
		}
		exitCtExpression(expression);
	}
	
	@Override
	public <T, A extends T> void visitCtOperatorAssignment(CtOperatorAssignment<T, A> assignment) {
		DependencyFlowNode nodeFromLastStatement=this.getLastNode();
		enterCtStatement(assignment);
		registerStatementLabel(assignment);
		enterCtExpression(assignment);

		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		scan(assignment.getAssigned());
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariablesAssigned = getInvolvedVariables(copybefore, copyafter);
        if(assignment.getAssigned() instanceof CtArrayAccess)
        	involvedVariablesAssigned.removeAll(arrayIndexVarsAssigned);
        dealVariablesInGraph(involvedVariablesAssigned,"Assigned", null, nodeFromLastStatement, false);

        // the operators like +=, *= are sent as one operator token
		result.getTokenSequence().add(Pair.of(ExtendedCDGBuilder.getOperatorText(assignment.getKind()) + "=",null));
		
		ArrayList<Pair<String, CtElement>> copybeforeAssignment = new ArrayList<>(result.getTokenSequence());
		scan(assignment.getAssignment());
		ArrayList<Pair<String, CtElement>> copyafterAssignment = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariablesAssignement = getInvolvedVariables(copybeforeAssignment, copyafterAssignment); 
		
    	if(assignment.getAssignment() instanceof CtConditional) {
    		variablesConditionalThen.addAll(variablesConditionalElse);
    		involvedVariablesAssignement = new ArrayList<>(variablesConditionalThen);
    	} else if(assignment.getAssignment() instanceof CtSwitchExpression) {
    		involvedVariablesAssignement.clear();
		}  else if(assignment.getAssignment() instanceof CtLambda) {
			involvedVariablesAssignement.clear();
		}  else if(assignment.getAssignment() instanceof CtNewClass) {
			involvedVariablesAssignement.clear();
		} 
        dealVariablesInGraph(involvedVariablesAssignement,"Assignment", null, nodeFromLastStatement, true);
    	
		exitCtExpression(assignment);
		exitCtStatement(assignment);	
	}
	
	@Override
	public void visitCtPackage(CtPackage ctPackage) { }

	public void writeImports(Collection<CtImport> imports) { }
	
	@Override
	public <T> void visitCtParameter(CtParameter<T> parameter) {
		this.writeAnnotations(parameter);
		this.writeModifiers(parameter);
		if (parameter.isVarArgs()) {
			scan(((CtArrayTypeReference<T>) parameter.getType()).getComponentType());
			result.getTokenSequence().add(Pair.of("...",null));
		} else if (parameter.isInferred()) {
			result.getTokenSequence().add(Pair.of("var",null));
		} else {
			scan(parameter.getType());
		}	
		result.getTokenSequence().add(Pair.of(parameter.getSimpleName(),parameter));
	}
	
	@Override
	public <T> void visitCtParameterReference(CtParameterReference<T> reference) {
		result.getTokenSequence().add(Pair.of(reference.getSimpleName(),reference));
	}
	
	@Override
	public <R> void visitCtReturn(CtReturn<R> returnStatement) {
		DependencyFlowNode nodeFromLastStatement=this.getLastNodeToUse();
		enterCtStatement(returnStatement);
		registerStatementLabel(returnStatement);

		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		result.getTokenSequence().add(Pair.of("return",null));
		int returnindex = result.getTokenSequence().size();

		scan(returnStatement.getReturnedExpression());
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
        if(returnStatement.getReturnedExpression() instanceof CtConditional) {
        	involvedVariables.clear();
		} else if(returnStatement.getReturnedExpression() instanceof CtLambda) {
			involvedVariables.clear();
		} else if(returnStatement.getReturnedExpression() instanceof CtNewClass) {
			involvedVariables.clear();
		} else if(returnStatement.getReturnedExpression() instanceof CtSwitchExpression) {
			involvedVariables.clear();
		}  else if(returnStatement.getReturnedExpression() instanceof CtInvocation) {
			involvedVariables.clear();
		}  else if(returnStatement.getReturnedExpression() instanceof CtConstructorCall) {
			involvedVariables.clear();
		} 
        DependencyFlowNode returnnode = new DependencyFlowNode(returnStatement, result,
				NodeKind.BRANCH, NodeType.Returnexit, NodeRole.BLOCKDEFAULT, null, returnindex, ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(returnnode);
        dealVariablesInGraph(involvedVariables,"others", returnnode, nodeFromLastStatement, false);
        if(involvedVariables.size()==0)
		    tryAddEdge(nodeFromLastStatement, returnnode);
		tryAddEdge(returnnode, exitNode);
		exitCtStatement(returnStatement);	
	}
	
	@Override
	public void visitCtStatementList(CtStatementList statements) {
		for (CtStatement s : statements.getStatements()) {
			scan(s);
		}
	}
	
	@Override
	public <E> void visitCtSwitch(CtSwitch<E> switchStatement) {
		DependencyFlowNode nodeFromLastStatement=this.getLastNodeToUse();
		registerStatementLabel(switchStatement);
		enterCtStatement(switchStatement);
	    DependencyFlowNode convergenceNode = new DependencyFlowNode(null, result, 
	      				NodeKind.BRANCH, NodeType.SwitchCONVERGE, NodeRole.BLOCKDEFAULT, null, result.getTokenSequence().size(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(convergenceNode);
	    breakingBad.push(convergenceNode);
	    
		outlevel++;
		int switchindex=0;
		result.getTokenSequence().add(Pair.of("switch",null));
		switchindex=result.getTokenSequence().size();
	//	result.getTokenSequence().add(Pair.of("(",null));
		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		scan(switchStatement.getSelector());
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
		DependencyFlowNode switchNode = new DependencyFlowNode (switchStatement.getSelector(), result, NodeKind.BRANCH, 
				NodeType.Switchselector, NodeRole.BLOCKDEFAULT, null, switchindex, ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(switchNode);
		dealVariablesInGraph(involvedVariables,"others", switchNode, nodeFromLastStatement, false);
		if(involvedVariables.size()==0)
		    tryAddEdge(nodeFromLastStatement, switchNode);
		
	//	result.getTokenSequence().add(Pair.of(")",null));
	//	result.getTokenSequence().add(Pair.of("{",null));
		lastControlNode.push(switchNode);
		for (CtCase<?> c : switchStatement.getCases()) {
			innerindex++;
			scan(c);
		}
		lastControlNode.pop();
	//	result.getTokenSequence().add(Pair.of("}",null));	
		tryAddEdge(this.getLastNodeToUse(), convergenceNode);
		breakingBad.pop();
		outlevel--;
		lastConverengenceNode = convergenceNode;
		exitCtStatement(switchStatement);
	}
	
	@Override
	public <T, S> void visitCtSwitchExpression(CtSwitchExpression<T, S> switchExpression) {
		DependencyFlowNode nodeFromLastStatement=this.getLastNodeToUse();
		enterCtExpression(switchExpression);
		outlevel++;
	    DependencyFlowNode convergenceNode = new DependencyFlowNode(null, result, 
	      				NodeKind.BRANCH, NodeType.SwitchCONVERGE, NodeRole.BLOCKDEFAULT, null, result.getTokenSequence().size(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(convergenceNode);
	    breakingBad.push(convergenceNode);
		
		int switchindex=0;
		result.getTokenSequence().add(Pair.of("switch",null));
		switchindex=result.getTokenSequence().size();
	//	result.getTokenSequence().add(Pair.of("(",null));
		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		scan(switchExpression.getSelector());
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
		DependencyFlowNode switchNode = new DependencyFlowNode (switchExpression.getSelector(), result, NodeKind.BRANCH, 
				NodeType.Switchselector, NodeRole.BLOCKDEFAULT, null, switchindex, ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(switchNode);
		dealVariablesInGraph(involvedVariables,"others", switchNode, nodeFromLastStatement, parentassignment(switchExpression));
		if(involvedVariables.size()==0)
		    tryAddEdge(nodeFromLastStatement, switchNode);
		
	//	result.getTokenSequence().add(Pair.of(")",null));
	//	result.getTokenSequence().add(Pair.of("{",null));
		lastControlNode.push(switchNode);
		for (CtCase<?> c : switchExpression.getCases()) {
			innerindex++;
			scan(c);
		}
		lastControlNode.pop();
	//	result.getTokenSequence().add(Pair.of("}",null));	
	    tryAddEdge(this.getLastNodeToUse(), convergenceNode);
	    if(switchExpression.getParent() instanceof CtAssignment && switchExpression.getRoleInParent() == CtRole.ASSIGNMENT)
			tryAddEdge(convergenceNode, nodeFromLastStatement, true, false);		   
		breakingBad.pop();
		outlevel--;
		exitCtExpression(switchExpression);
	}
	
	@Override
	public void visitCtSynchronized(CtSynchronized synchro) {
		DependencyFlowNode nodeFromLastStatement=this.getLastNodeToUse();
		enterCtStatement(synchro);
		registerStatementLabel(synchro);
		result.getTokenSequence().add(Pair.of("synchronized",null));
		int synchronizedindex = result.getTokenSequence().size();

		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		if (synchro.getExpression() != null) {
		//	result.getTokenSequence().add(Pair.of("(",null));
			scan(synchro.getExpression());
		//	result.getTokenSequence().add(Pair.of(")",null));
		}
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
		DependencyFlowNode synchronizedNode = new DependencyFlowNode (synchro, result, NodeKind.BRANCH, 
				NodeType.Synchronizedentry, NodeRole.BLOCKDEFAULT, null, synchronizedindex, ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(synchronizedNode);
        if(involvedVariables.size()>0)
            dealVariablesInGraph(involvedVariables,"others", synchronizedNode, nodeFromLastStatement, false);
        else tryAddEdge(nodeFromLastStatement, synchronizedNode);
		scan(synchro.getBlock());
		exitCtStatement(synchro);
	}
	
	@Override
	public void visitCtThrow(CtThrow throwStatement) {
		enterCtStatement(throwStatement);
		result.getTokenSequence().add(Pair.of("throw",null));
		int throwindex = result.getTokenSequence().size();
		scan(throwStatement.getThrownExpression());
		if (exceptionControlFlowTactic != null) {
			exceptionControlFlowTactic.handleThrowStatement(this, throwStatement, throwindex);
		}
		exitCtStatement(throwStatement);
	}
	
	@Override
	public void visitCtTry(CtTry tryBlock) {
		enterCtStatement(tryBlock);
		result.getTokenSequence().add(Pair.of("try",null));
		int tryindex = result.getTokenSequence().size();		
		if (exceptionControlFlowTactic != null) {
			exceptionControlFlowTactic.handleTryStatement(this, tryBlock, tryindex);
		}	
		exitCtStatement(tryBlock);
	}
	
	@Override
	public void visitCtTryWithResource(CtTryWithResource tryWithResource) {
		enterCtStatement(tryWithResource);
		result.getTokenSequence().add(Pair.of("try",null));
		int tryindex = result.getTokenSequence().size();	
		if (exceptionControlFlowTactic != null) {
			exceptionControlFlowTactic.handleTryStatement(this, tryWithResource, tryindex);
		}		
		exitCtStatement(tryWithResource);
	}
	
	@Override
	public void visitCtTypeParameterReference(CtTypeParameterReference ref) {
		if (ref.isImplicit()) {
			return;
		}
		this.writeAnnotations(ref);
		if (printQualified(ref)) {
			result.getTokenSequence().add(Pair.of(ref.getQualifiedName(),null));
		} else {
			result.getTokenSequence().add(Pair.of(ref.getSimpleName(),null));
		}
	}
	
	@Override
	public void visitCtWildcardReference(CtWildcardReference wildcardReference) {
		if (wildcardReference.isImplicit()) {
			return;
		}
		this.writeAnnotations(wildcardReference);
		result.getTokenSequence().add(Pair.of("?",null));
		// we ignore printing "extends Object" except if it's explicit
		if (!wildcardReference.isDefaultBoundingType() || !wildcardReference.getBoundingType().isImplicit()) {
			if (wildcardReference.isUpper()) {
				result.getTokenSequence().add(Pair.of("extends",null));
			} else {
				result.getTokenSequence().add(Pair.of("super",null));
			}
			scan(wildcardReference.getBoundingType());
		}
	}
	
	@Override
	public <T> void visitCtIntersectionTypeReference(CtIntersectionTypeReference<T> reference) {
		if (reference.isImplicit()) {
			return;
		}
		this.printList(reference.getBounds(),
			null, null, "", null, bound -> scan(bound));
//		this.printList(reference.getBounds(),
//				null, null, "&", null, bound -> scan(bound));
	}
	
	@Override
	public <T> void visitCtTypeAccess(CtTypeAccess<T> typeAccess) {
		if (!ignoreImplicit && typeAccess.isImplicit()) {
			return;
		}
		enterCtExpression(typeAccess);
		scan(typeAccess.getAccessedType());
		exitCtExpression(typeAccess);
	}
	
	@Override
	public <T> void visitCtUnaryOperator(CtUnaryOperator<T> operator) {
		DependencyFlowNode nodeFromLastStatement=this.getLastNode();
		enterCtStatement(operator);
		enterCtExpression(operator);
		UnaryOperatorKind op = operator.getKind();
		if (ExtendedCDGBuilder.isPrefixOperator(op)) {
			result.getTokenSequence().add(Pair.of(ExtendedCDGBuilder.getOperatorText(op),null));
		}
		
		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		scan(operator.getOperand());
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
        checkisAssignmenrExper(operator);
        dealVariablesInGraph(involvedVariables,"others", null, nodeFromLastStatement, parentassignment(operator));
		
		if (ExtendedCDGBuilder.isSufixOperator(op)) {
			result.getTokenSequence().add(Pair.of(ExtendedCDGBuilder.getOperatorText(op),null));
		}
		isExperAssignmentVars = false;
		exitCtExpression(operator);
		exitCtStatement(operator);
	}
	
	public static boolean isPrefixOperator(UnaryOperatorKind o) {
		return isSufixOperator(o) == false;
	}
	
	public static boolean isSufixOperator(UnaryOperatorKind o) {
		return o.name().startsWith("POST");
	}
	
	@Override
	public <T> void visitCtVariableRead(CtVariableRead<T> variableRead) {
		enterCtExpression(variableRead);
		result.getTokenSequence().add(Pair.of(variableRead.getVariable().getSimpleName(),variableRead));
		exitCtExpression(variableRead);
	}
	
	@Override
	public <T> void visitCtVariableWrite(CtVariableWrite<T> variableWrite) {
		enterCtExpression(variableWrite);
		result.getTokenSequence().add(Pair.of(variableWrite.getVariable().getSimpleName(),variableWrite));
		exitCtExpression(variableWrite);
	}
	
	@Override
	public void visitCtWhile(CtWhile whileLoop) {
		DependencyFlowNode nodeFromLastStatement=this.getLastNodeToUse();
		enterCtStatement(whileLoop);
		registerStatementLabel(whileLoop);

		DependencyFlowNode convergenceNode = new DependencyFlowNode(null, result, NodeKind.BRANCH, 
				NodeType.WhileCONVERGE, NodeRole.BLOCKDEFAULT, null, result.getTokenSequence().size(), ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(convergenceNode);
		breakingBad.push(convergenceNode);
		
		result.getTokenSequence().add(Pair.of("while",null));
	//	result.getTokenSequence().add(Pair.of("(",null));	
		int whileindex = result.getTokenSequence().size();
		
		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
		scan(whileLoop.getLoopingExpression());
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
        DependencyFlowNode whilecontinue = new DependencyFlowNode(whileLoop.getLoopingExpression(), result, 
				NodeKind.BRANCH, NodeType.Whilecontinue, NodeRole.BLOCKDEFAULT, null, whileindex, ExtendedCDGBuilder.outlevel, ExtendedCDGBuilder.innerindex);
		result.addVertex(whilecontinue);
		continueBad.push(whilecontinue);
        dealVariablesInGraph(involvedVariables,"others", whilecontinue, nodeFromLastStatement, false);
        if(involvedVariables.size()==0)
		    tryAddEdge(nodeFromLastStatement, whilecontinue);

	//	result.getTokenSequence().add(Pair.of(")",null));
		int idbefore=this.getLastNode().getId();
		this.writeIfOrLoopBlock(whileLoop.getBody());
		int idafter=this.getLastNode().getId();
		if(idafter>idbefore)	
		    tryAddEdge(this.getLastNodeToUse(), whilecontinue, true, false);
			
		tryAddEdge(whilecontinue, convergenceNode);
		breakingBad.pop();
		continueBad.pop();	
		lastConverengenceNode = convergenceNode;
		exitCtStatement(whileLoop);
	}
	
	@Override
	public <T> void visitCtCodeSnippetExpression(CtCodeSnippetExpression<T> expression) { }
	
	@Override
	public void visitCtCodeSnippetStatement(CtCodeSnippetStatement statement) { }
	
	@Override
	public <T> void visitCtUnboundVariableReference(CtUnboundVariableReference<T> reference) {
		result.getTokenSequence().add(Pair.of(reference.getSimpleName(),reference));
	}
		
	@Override
	public void visitCtYieldStatement(CtYieldStatement statement) {
		if (statement.isImplicit()) {
			DependencyFlowNode nodeFromLastStatement=this.getLastNode();
			ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
			scan(statement.getExpression());
			ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
	        dealVariablesInGraph(getInvolvedVariables(copybefore, copyafter),"others", null, nodeFromLastStatement, false);
			exitCtStatement(statement);
			return;
		}
		DependencyFlowNode nodeFromLastStatement=this.getLastNode();
		enterCtStatement(statement);
		result.getTokenSequence().add(Pair.of("yield",null));
		ArrayList<Pair<String, CtElement>> copybeforewhole = new ArrayList<>(result.getTokenSequence());
		scan(statement.getExpression());
		ArrayList<Pair<String, CtElement>> copyafterwhole = new ArrayList<>(result.getTokenSequence());
        dealVariablesInGraph(getInvolvedVariables(copybeforewhole, copyafterwhole),"others", null, nodeFromLastStatement, false);
		exitCtStatement(statement);
	}
	
	@Override
	public void visitCtTypePattern(CtTypePattern pattern) {
		enterCtExpression(pattern);
		scan(pattern.getVariable());
		exitCtExpression(pattern);
	}
	
	@Override
	public void visitCtRecord(CtRecord recordType) {
		DependencyFlowNode nodeFromLastStatement=this.getLastNode();
		this.pushCurrentThis(recordType);
		visitCtType(recordType);
		result.getTokenSequence().add(Pair.of("record",null));
		result.getTokenSequence().add(Pair.of(stripLeadingDigits(recordType.getSimpleName()),null));
		this.writeFormalTypeParameters(recordType);
		
		ArrayList<Pair<String, CtElement>> copybefore = new ArrayList<>(result.getTokenSequence());
	//	this.printList(recordType.getRecordComponents(), null, "(", ",", ")", this::visitCtRecordComponent);
		this.printList(recordType.getRecordComponents(), null, "", "", "", this::visitCtRecordComponent);
		ArrayList<Pair<String, CtElement>> copyafter = new ArrayList<>(result.getTokenSequence());
        ArrayList<Pair<CtElement, Integer>> involvedVariables = getInvolvedVariables(copybefore, copyafter);
        dealVariablesInGraph (involvedVariables,"others", null, nodeFromLastStatement, false);
        
		this.writeImplementsClause(recordType);
	//	result.getTokenSequence().add(Pair.of("{",null));
		this.writeElementList(recordType.getTypeMembers());
	//	result.getTokenSequence().add(Pair.of("}",null));
		this.popCurrentThis();
	}
	
	@Override
	public void visitCtRecordComponent(CtRecordComponent recordComponent) {
		this.writeAnnotations(recordComponent);
		visitCtTypeReference(recordComponent.getType());
		result.getTokenSequence().add(Pair.of(recordComponent.getSimpleName(),recordComponent));
	}
	
	public static String getOperatorText(UnaryOperatorKind o) {
		switch (o) {
			case POS:
				return "+";
			case NEG:
				return "-";
			case NOT:
				return "!";
			case COMPL:
				return "~";
			case PREINC:
				return "++";
			case PREDEC:
				return "--";
			case POSTINC:
				return "++";
			case POSTDEC:
				return "--";
			default:
				throw new SpoonException("Unsupported operator " + o.name());
		}
	}
	
	protected void enter(CtElement e) { }

	protected void exit(CtElement e) { }
	
	private void visitCtTypeReferenceWithoutGenerics(CtTypeReference<?> ref) {
		visitCtTypeReference(ref, false);
	}
	
	private boolean shouldPrintTarget(CtExpression<?> target) {
		if (target == null) {
			return false;
		}
		if (!target.isImplicit()) {
			return true;
		}
		if (!ignoreImplicit) {
			return false;
		}
		if (target instanceof CtThisAccess) {
			return false;
		}
		return true;
	}
	
	public void writeIfOrLoopBlock(CtStatement block) {
		if (block != null) {
			writeStatement(block);
		} else {
			result.getTokenSequence().add(Pair.of(";",null));
		}
	}
	
	public void writeExecutableParameters(CtExecutable<?> executable) {
//		printList(executable.getParameters(), null,
//			"(",  ",", ")", p -> scan(p));
		printList(executable.getParameters(), null,
				"",  "", "", p -> scan(p));
	}
	
	public void writeThrowsClause(CtExecutable<?> executable) {
		if (!executable.getThrownTypes().isEmpty()) {
//			printList(executable.getThrownTypes(), "throws",
//				null, ",", null, ref -> scan(ref));
			printList(executable.getThrownTypes(), "throws",
					null, "", null, ref -> scan(ref));
		}
	}
	
	public void writeFormalTypeParameters(CtFormalTypeDeclarer ctFormalTypeDeclarer) {
		final Collection<CtTypeParameter> parameters = ctFormalTypeDeclarer.getFormalCtTypeParameters();
		if (parameters == null) {
			return;
		}
		if (!parameters.isEmpty()) {
	//		printList(parameters, null, "<", ",", ">", parameter -> scan(parameter));
			printList(parameters, null, "", "", "", parameter -> scan(parameter));
		}
	}
	
	public void writeExtendsClause(CtType<?> type) {
		if (type.getSuperclass() != null) {
			result.getTokenSequence().add(Pair.of("extends",null));
			this.scan(type.getSuperclass());
		}
	}
	
	public void writeImplementsClause(CtType<?> type) {
		if (!type.getSuperInterfaces().isEmpty()) {
//			printList(type.getSuperInterfaces(), "implements",
//			   null, ",", null, ref -> scan(ref));
			printList(type.getSuperInterfaces(), "implements",
					   null, "", null, ref -> scan(ref));
		}
	}
	
	public static String getOperatorText(BinaryOperatorKind o) {
		switch (o) {
			case OR:
				return "||";
			case AND:
				return "&&";
			case BITOR:
				return "|";
			case BITXOR:
				return "^";
			case BITAND:
				return "&";
			case EQ:
				return "==";
			case NE:
				return "!=";
			case LT:
				return "<";
			case GT:
				return ">";
			case LE:
				return "<=";
			case GE:
				return ">=";
			case SL:
				return "<<";
			case SR:
				return ">>";
			case USR:
				return ">>>";
			case PLUS:
				return "+";
			case MINUS:
				return "-";
			case MUL:
				return "*";
			case DIV:
				return "/";
			case MOD:
				return "%";
			case INSTANCEOF:
				return "instanceof";
			default:
				throw new SpoonException("Unsupported operator " + o.name());
		}
	}
	
	protected void enterCtStatement(CtStatement s) {
		if(!lastdealedNodes.isEmpty()) {
			lastdealedNonEmptyNodes.clear();
			lastdealedNonEmptyNodes.addAll(lastdealedNodes);
		}
		lastdealedNodes.clear(); 		
		lastlastnodetouse = this.getLastNodeToUse();
		if (this.NEXT_FOR_VARIABLE) {
			this.writeAnnotations(s);
		}
		if (!this.FIRST_FOR_VARIABLE && !this.NEXT_FOR_VARIABLE) {
			if (s.getLabel() != null) {
				result.getTokenSequence().add(Pair.of(s.getLabel(),null));
				result.getTokenSequence().add(Pair.of(":",null));
			}
		}
	}
	
	protected void exitCtStatement(CtStatement statement) {
		if (!(statement instanceof CtBlock || statement instanceof CtIf || statement instanceof CtFor || 
				statement instanceof CtForEach || statement instanceof CtWhile || statement instanceof CtTry
				|| statement instanceof CtSwitch || statement instanceof CtSynchronized || statement instanceof CtClass
				|| statement instanceof CtComment || statement.getParent() instanceof CtForEach)) {
			
			if (!this.FIRST_FOR_VARIABLE && !this.NEXT_FOR_VARIABLE) {			
				if(statement instanceof CtInvocation || statement instanceof CtConstructorCall || statement instanceof CtNewClass || statement instanceof CtOperatorAssignment
						|| statement instanceof CtUnaryOperator || statement instanceof CtAssignment) {
					if (statement.getParent() instanceof CtBlock || statement.getParent() instanceof CtCase)
						result.getTokenSequence().add(Pair.of(";", null));
				}  else	result.getTokenSequence().add(Pair.of(";", null));
			}
		}
		
		if(statement instanceof CtDo || statement instanceof CtFor || statement instanceof CtForEach ||
			statement instanceof CtIf || statement instanceof CtSwitch || statement instanceof CtWhile 
			|| statement instanceof CtTry || statement instanceof CtTryWithResource) {
			
		} else lastConverengenceNode = null;
	}
	
	protected void exitCtExpression(CtExpression<?> e) {
		while ((!this.parenthesedExpression.isEmpty()) && e == this.parenthesedExpression.peek()) {
			this.parenthesedExpression.pop();
		//	result.getTokenSequence().add(Pair.of(")",null));
		}
	}
	
	protected void enterCtExpression(CtExpression<?> e) {		
		if (shouldSetBracket(e)) {
			this.parenthesedExpression.push(e);
		//	result.getTokenSequence().add(Pair.of("(",null));
		}
		if (!e.getTypeCasts().isEmpty()) {
			for (CtTypeReference<?> r : e.getTypeCasts()) {
		//		result.getTokenSequence().add(Pair.of("(",null));
				scan(r);
		//		result.getTokenSequence().add(Pair.of(")",null));
		//		result.getTokenSequence().add(Pair.of("(",null));
				this.parenthesedExpression.push(e);
			}
		}
		lastConverengenceNode = null;
	}
	
	private boolean shouldSetBracket(CtExpression<?> e) {
		if (!e.getTypeCasts().isEmpty()) {
			return true;
		}
		try {
			if ((e.getParent() instanceof CtBinaryOperator) || (e.getParent() instanceof CtUnaryOperator)) {
				return (e instanceof CtAssignment) || (e instanceof CtConditional) || (e instanceof CtUnaryOperator) || e instanceof CtBinaryOperator;
			}
			if (e.getParent() instanceof CtTargetedExpression && ((CtTargetedExpression<?, ?>) e.getParent()).getTarget() == e) {
				return (e instanceof CtBinaryOperator) || (e instanceof CtAssignment) || (e instanceof CtConditional) || (e instanceof CtUnaryOperator);
			}
		} catch (ParentNotInitializedException ex) {
			// nothing we accept not to have a parent
		}
		return false;
	}
	
	public void writeElementList(List<CtTypeMember> elements) {
		for (CtTypeMember element : elements) {
			if (!element.isImplicit()) 
				this.scan(element);
		}
	}
	
	private <T> void visitCtType(CtType<T> type) {
		if (type.isTopLevel()) {
			this.currentTopLevel = type;
		}
		this.visitCtNamedElement(type);
		this.writeModifiers(type);
	}
	
	public void visitCtNamedElement(CtNamedElement namedElement) {
		writeAnnotations(namedElement);
	}
	
	public void writeModifiers(CtModifiable modifiable) {
		List<String> firstPosition = new ArrayList<>(); // visibility: public, private, protected
		List<String> secondPosition = new ArrayList<>(); // keywords: static, abstract
		List<String> thirdPosition = new ArrayList<>(); // all other things

		for (CtExtendedModifier extendedModifier : modifiable.getExtendedModifiers()) {
			if (!extendedModifier.isImplicit()) {
				ModifierKind modifierKind = extendedModifier.getKind();
				if (modifierKind == ModifierKind.PUBLIC || modifierKind == ModifierKind.PRIVATE || modifierKind == ModifierKind.PROTECTED) {
					firstPosition.add(modifierKind.toString());
				} else if (modifierKind == ModifierKind.ABSTRACT || modifierKind == ModifierKind.STATIC) {
					secondPosition.add(modifierKind.toString());
				} else {
					thirdPosition.add(modifierKind.toString());
				}
			}
		}

		for (String s : firstPosition) {
			result.getTokenSequence().add(Pair.of(s,null));
		}

		for (String s : secondPosition) {
			result.getTokenSequence().add(Pair.of(s,null));
		}

		for (String s : thirdPosition) {
			result.getTokenSequence().add(Pair.of(s,null));
		}

		if (modifiable instanceof CtMethod) {
			CtMethod<?> m = (CtMethod<?>) modifiable;
			if (m.isDefaultMethod()) {
				result.getTokenSequence().add(Pair.of("default",null));
			}
		}
	}
}
