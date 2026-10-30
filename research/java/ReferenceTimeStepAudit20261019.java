import com.comsol.model.*;import com.comsol.model.util.*;import java.io.*;import java.util.*;
/** Read-only solver-output and nonlinear-tolerance audit. Never saves a model. */
public class ReferenceTimeStepAudit20261019 {
 static void inspect(SolverFeature f,String path,PrintWriter out){
  out.println("FEATURE|"+path+"|"+f.getType());
  for(String key:f.properties())if(key.toLowerCase().matches(".*(step|tout|store|tol|order|consistent|damp|seg|jac).*")){
   try{out.println(path+"|"+key+"|"+Arrays.toString(f.getStringArray(key)));}
   catch(Throwable e){try{out.println(path+"|"+key+"|"+f.getString(key));}catch(Throwable ignored){}}
  }
  for(String tag:f.feature().tags())inspect(f.feature(tag),path+"/"+tag,out);
 }
 public static void main(String[] a){try{
  ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(false);
  Model m=ModelUtil.load("TimeStepAudit",a[0]);PrintWriter out=new PrintWriter(a[1],"UTF-8");
  out.println("MODEL|"+a[0]);out.println("SAVED_TIMES|"+Arrays.toString(m.sol("refTime").getPVals()));
  inspect(m.sol("refTime").feature("t1"),"refTime/t1",out);out.close();
  System.out.println("READ_ONLY_TIME_STEP_AUDIT_COMPLETE");System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
