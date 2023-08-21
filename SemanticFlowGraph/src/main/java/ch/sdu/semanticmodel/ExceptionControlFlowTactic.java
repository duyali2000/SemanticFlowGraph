package ch.sdu.semanticmodel;

import spoon.reflect.code.CtThrow;
import spoon.reflect.code.CtTry;

public interface ExceptionControlFlowTactic {

	void handleTryStatement(ExtendedCDGBuilder builder, CtTry tryBlock, int tryindex);
	
	void handleThrowStatement(ExtendedCDGBuilder builder, CtThrow throwStatement, int indexForThrow);

	void postProcess(ExtendedCodeDependencyGraph graph);
}
