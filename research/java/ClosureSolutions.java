import com.comsol.model.*;import com.comsol.model.util.*;import java.util.*;
public class ClosureSolutions{
 public static void main(String[] a)throws Exception{
  ModelUtil.initStandalone(false);Model m=ModelUtil.load("Solutions",a[0]);
  m.result().numerical().create("solutionProbe","Interp");m.result().numerical("solutionProbe").set("coord",new double[][]{{.05},{.00105}});
  for(String t:m.result().dataset().tags()){
   System.out.println("DATASET|"+t+"|"+m.result().dataset(t).getType());
   for(String p:m.result().dataset(t).properties())try{System.out.println("PROPERTY|"+p+"|"+Arrays.toString(m.result().dataset(t).getStringArray(p)));}catch(Throwable e){}
   try{m.result().numerical("solutionProbe").set("data",t);m.result().numerical("solutionProbe").set("expr",new String[]{"T","V_cell"});System.out.println("VALUES|"+t+"|"+Arrays.deepToString(m.result().numerical("solutionProbe").getData()));}catch(Throwable e){System.out.println("ERROR|"+t+"|"+e.getMessage());}
  }
  System.out.println("MESH_VERTICES|"+m.mesh("mesh1").getVertex().length);
  for(String t:m.sol().tags())for(String f:m.sol(t).feature().tags())System.out.println("SOL_FEATURE|"+t+"|"+f+"|"+m.sol(t).feature(f).getType());
  ModelUtil.remove("Solutions");System.exit(0);
 }
}
