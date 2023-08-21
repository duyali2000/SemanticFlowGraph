package ch.sdu.semanticmodel;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.EnumSet;
import java.util.Scanner;

import spoon.Launcher;
import spoon.reflect.CtModel;
import spoon.reflect.declaration.CtMethod;
import spoon.reflect.factory.TypeFactory;
import spoon.reflect.visitor.filter.TypeFilter;
import spoon.support.compiler.VirtualFile;

public class Activator {

	public static void main(String[] args) {
		
		if(args.length!=2) {
		     System.out.println("Usage: give the path for the method snippets that we will generate SFG (first argument) and "
		     		+ "the path for storing the generated SFG (second argument)");
		     return;
		}     
		
		File[] files = new File(args[0]).listFiles();
		for (File file : files) {
		    if (file.isFile()) {	
				String statement="";
				statement+="class Wrapper {";
				statement+="\n";
				Scanner scanner = null;
		        try {
					scanner = new Scanner(file, "UTF-8");
					while (scanner.hasNextLine()) {
						   String line = scanner.nextLine();
						   statement+= deal(line);   
						   statement+="\n";
					}
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				}
				statement+="}";
				statement+="\n";
				int fileindex= Integer.valueOf(file.getName().replace(".txt", ""));
                try {
				Launcher launcher = new Launcher();
				launcher.addInputResource(new VirtualFile(statement));
				launcher.getEnvironment().setNoClasspath(true);
				launcher.getEnvironment().setAutoImports(true); 
				CtModel partialmodel = launcher.buildModel();	

				CtMethod<?> next= partialmodel.getElements(new TypeFilter<>(CtMethod.class)).iterator().next(); 	
				
				ExtendedCodeDependencyGraph graph=buildGraphForMethod(next, launcher.getFactory().Type());	

				graph.toGraphVisText(fileindex, args[1]);
                } catch (Exception e) {
        			System.out.println(e.getMessage());
        		}
		    }
		}
	}
	
	public static String deal(String s) { // remove chinese characters
        StringBuffer sb = new StringBuffer(s);
        StringBuffer se = new StringBuffer();    
        int l = sb.length();
        char c;
        for (int i = 0; i < l; i++) {
            c = sb.charAt(i);                   
            if (Character.UnicodeScript.of(c) != Character.UnicodeScript.HAN) {
                se.append(c);
            }
        }
        return new String(se);
    }
	
	public static ExtendedCodeDependencyGraph buildGraphForMethod(CtMethod<?> method, TypeFactory typefactory) {
		ExtendedCDGBuilder builder = new ExtendedCDGBuilder();
		EnumSet<NaiveExceptionControlFlowTactic.Options> options;
	    options = EnumSet.of(NaiveExceptionControlFlowTactic.Options.ReturnWithoutFinalizers);
		builder.setExceptionControlFlowTactic(new NaiveExceptionControlFlowTactic(options));
		ExtendedCodeDependencyGraph graph = builder.build(method, typefactory);
		graph.simplify();
		
		return graph;
	}
}
