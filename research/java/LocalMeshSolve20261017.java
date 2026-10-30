import com.comsol.model.*;import com.comsol.model.util.*;import java.io.*;import java.util.*;
public class LocalMeshSolve20261017{
 public static void main(String[] a){try{
 ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(a[2]);Model m=ModelUtil.load("LocalMeshStudy",a[0]);
 m.sol("sol1").copySolution("localMeshWarm17");
 int before=m.mesh("mesh1").getVertex()[0].length;
 m.mesh("mesh1").create("localAdapt17","Adapt");
 m.mesh("mesh1").feature("localAdapt17").set("method","regular");
 m.mesh("mesh1").feature("localAdapt17").set("exprtype","size");
 m.mesh("mesh1").feature("localAdapt17").set("solution","none");
 m.mesh("mesh1").feature("localAdapt17").set("maxrefinement",1);
 m.mesh("mesh1").feature("localAdapt17").set("sizeexpr","if("+a[3]+",0.5*h,h)");
 m.mesh("mesh1").run();
 double[][] xy=m.mesh("mesh1").getVertex();
 System.out.println("LOCAL_MESH_BUILT|before="+before+"|after="+xy[0].length+"|mask="+a[3]);System.out.flush();
 PrintWriter p=new PrintWriter(a[1].replace(".mph","_mesh.csv"),"UTF-8");p.println("x,y");for(int i=0;i<xy[0].length;i++)p.printf(Locale.US,"%.17g,%.17g%n",xy[0][i],xy[1][i]);p.close();
 if(xy[0].length<=before)throw new RuntimeException("Refinement added no vertices");
 if(xy[0].length>700000)throw new RuntimeException("Predeclared mesh memory guard exceeded:700000 vertices");
 m.sol("sol1").feature("v1").set("initmethod","sol");m.sol("sol1").feature("v1").set("initsol","localMeshWarm17");m.sol("sol1").feature("v1").set("solnum","1");
 m.sol("sol1").feature("s1").set("stol","1e-9");m.sol("sol1").feature("s1").feature("fc1").set("initstep","1");m.sol("sol1").feature("s1").feature("fc1").set("maxiter","40");
 for(String tag:m.sol("sol1").feature("s1").feature().tags())if(m.sol("sol1").feature("s1").feature(tag).getType().equals("Direct")){m.sol("sol1").feature("s1").feature(tag).set("ooc","on");m.sol("sol1").feature("s1").feature(tag).set("incore","manual");m.sol("sol1").feature("s1").feature(tag).set("oocmemory",4096);}
 m.label("Local regular mesh refinement; frozen physical settings; acceptance pending");m.save(a[1].replace(".mph","_configured.mph"));
 if(a.length>4&&a[4].equals("build")){System.out.println("BUILD_ONLY_COMPLETE");System.exit(0);}
 System.out.println("LOCAL_SOLVE_START");System.out.flush();m.sol("sol1").runAll();m.save(a[1]);m.save(a[1].replace(".mph","_history.java"),"java");System.out.println("LOCAL_SOLVE_COMPLETE");System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
