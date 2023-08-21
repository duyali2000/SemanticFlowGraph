package ch.sdu.semanticmodel;

import spoon.reflect.declaration.CtElement;

public class DependencyFlowNode {

	public static int count = 0;

	private int id;
	
	private NodeKind kind;
	
	private NodeType type;
	
	private NodeRole role;
				
	private int tokenindex=0;
	
    private int outlevel=0;
	
	private int innerindex=0;
	
	private Variable associatedVar;

	ExtendedCodeDependencyGraph parent;
	
	CtElement statement;
	
	public int getTokenIndex() {
		return tokenindex;
	}

	public void setTokenIndex(int tokenindex) {
		this.tokenindex = tokenindex;
	}
	
	public Variable getVariable() {
		return this.associatedVar;
	}

	public void setVariable(Variable associatedVar) {
		this.associatedVar = associatedVar;
	}

	public NodeKind getKind() {
		return kind;
	}

	public void setKind(NodeKind kind) {
		this.kind = kind;
	}
	
	public NodeType getType() {
		return type;
	}

	public void setType(NodeType type) {
		this.type = type;
	}
	
	public NodeRole getRole() {
		return role;
	}

	public void setRole(NodeRole role) {
		this.role = role;
	}
	
	public CtElement getStatement() {
		return statement;
	}

	public void setStatement(CtElement statement) {
		this.statement = statement;
	}

	public ExtendedCodeDependencyGraph getParent() {
		return this.parent;
	}

	public void setParent(ExtendedCodeDependencyGraph parent) {
		this.parent = parent;
	}

	public DependencyFlowNode (CtElement statement, ExtendedCodeDependencyGraph parent, NodeKind kind,
			NodeType type, NodeRole role, Variable associatedVar, int index, int outlevel, int innerindex) {
		this.kind = kind;
		this.parent = parent;
		this.statement = statement;
		this.type = type;
		this.role = role;
		this.tokenindex = index;
		this.associatedVar = associatedVar;
		++count;
		id = count;
		this.outlevel = outlevel;
		this.innerindex = innerindex;
	}
	
	public int getOutLevel() {
		return this.outlevel;
	}

	public void setOutLevel(int outlevel) {
		this.outlevel = outlevel;
	}
	
	public int getInnerIndex() {
		return this.innerindex;
	}

	public void setInnerIndex(int innerindex) {
		this.innerindex = innerindex;
	}

	public int getId() {
		return id;
	}

	@Override
	public String toString() {
		if (statement != null) {
			return id + " - " + statement.toString();
		} else {
			return kind + "_" + id;
		}
	}
}
