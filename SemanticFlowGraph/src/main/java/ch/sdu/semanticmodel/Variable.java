package ch.sdu.semanticmodel;

import spoon.reflect.declaration.CtElement;

public class Variable {

	private VariableType vartype;
	
	private VariableUseType varusetype;
	
	private int tokenindex=0;
	
	private String shortname;
	
	private CtElement associatedElement;
	
	public Variable (VariableType vartype, VariableUseType varusetype,  int tokenindex,
			String shortname, CtElement associatedElement) {
		this.vartype = vartype;
		this.varusetype = varusetype;
		this.tokenindex = tokenindex;
		this.shortname = shortname;
		this.associatedElement = associatedElement;
	}
	
	public CtElement getAssociatedElement() {
		return this.associatedElement;
	}

	public void setAssociatedElement (CtElement associatedElement) {
		this.associatedElement = associatedElement;
	}
	
	public String getShortName() {
		return this.shortname;
	}

	public void setShortName (String shortname) {
		this.shortname = shortname;
	}
	
	public int getTokenIndex() {
		return this.tokenindex;
	}

	public void setTokenIndex(int tokenindex) {
		this.tokenindex = tokenindex;
	}
	
	public VariableUseType getVariableUseType() {
		return this.varusetype;
	}

	public void setVariableUseType(VariableUseType varusetype) {
		this.varusetype = varusetype;
	}
	
	public VariableType getVariableType() {
		return this.vartype;
	}

	public void setVariableType(VariableType vartype) {
		this.vartype = vartype;
	}
}
