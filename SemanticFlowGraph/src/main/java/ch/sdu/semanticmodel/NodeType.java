package ch.sdu.semanticmodel;

public enum NodeType {
	TRY,         
	CATCH,       
	FINALLY,     
	ConditionalCONVERGE,   // 
	DoInsideCONVERGE,
	DoOutsideCONVERGE,
	ForCONVERGE,
	ForEachCONVERGE,
	IfCONVERGE,
	SwitchCONVERGE,
	WhileCONVERGE,
	TryCONVERGE,
	EXIT,       // 
	BEGIN,    //
	Branchbreak,
	Branchcontinue,
	Switchselector,
	CaseExpression,
	Conditionalcondition,
	ConditionalThen,
	ConditionalElse,
	Dowhilecondition,
	Forcondition,
	Foreachcondition,
	Ifcondition,
	IfThen, //
	IfElse, 
	Lambdaentry, 
	Lambdaexit, //
	Newclassentry, 
	Newclassexit, //
	Returnexit,
	Synchronizedentry,
	Whilecontinue,
	Exceptionthrow,
	INTEGER,
	FLOAT,
	LONG,
	DOUBLE,
	VOID,
	BOOLEAN,
	BYTE,
	CHARACTER,
	SHORT,
	PrimitiveOthers,
	DATE,
	ITERABLE,
	COLLECTION,
	LIST,
	SET,
	MAP,
	STRING,
	OtherJDKType,
	UserDefinedType,
	NullType
}
