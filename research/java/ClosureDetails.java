import com.comsol.model.*;import com.comsol.model.util.*;import com.comsol.model.physics.*;import java.util.*;
public class ClosureDetails{
 static void feat(PhysicsFeature f,String path){
  System.out.println("FEATURE|"+path+"|"+f.getType()+"|active="+f.isActive()+"|"+Arrays.toString(f.selection().entities()));
  for(String p:f.properties())try{System.out.println(path+"|"+p+"|"+Arrays.deepToString(f.getStringMatrix(p)));}catch(Throwable e){try{System.out.println(path+"|"+p+"|"+Arrays.toString(f.getStringArray(p)));}catch(Throwable ee){try{System.out.println(path+"|"+p+"|"+f.getString(p));}catch(Throwable eee){}}}
  for(String t:f.feature().tags())feat(f.feature(t),path+"/"+t);
 }
 public static void main(String[] a)throws Exception{
  ModelUtil.initStandalone(false);Model m=ModelUtil.load("Details",a[0]);
  for(String ph:m.physics().tags()){
   for(String t:m.physics(ph).feature().tags())feat(m.physics(ph).feature(t),ph+"/"+t);
   for(String t:m.physics(ph).prop().tags())for(String p:m.physics(ph).prop(t).properties())try{System.out.println("PROP|"+ph+"|"+t+"|"+p+"|"+Arrays.toString(m.physics(ph).prop(t).getStringArray(p)));}catch(Throwable e){}
  }
  for(String t:m.mesh("mesh1").feature().tags()){
   System.out.println("MESH|"+t+"|"+m.mesh("mesh1").feature(t).getType());
   for(String p:m.mesh("mesh1").feature(t).properties())try{System.out.println("MESH|"+t+"|"+p+"|"+Arrays.toString(m.mesh("mesh1").feature(t).getStringArray(p)));}catch(Throwable e){}
  }
  m.result().numerical().create("lineprobe","IntLine");m.result().numerical("lineprobe").set("data","dset1");m.result().numerical("lineprobe").set("solnum",6);
  for(int b:new int[]{2,4,6,8,10,11}){
   m.result().numerical("lineprobe").selection().set(new int[]{b});
   for(String e:new String[]{"up(dom)","down(dom)","ht.ndflux","ht.ndflux_u","ht.ndflux_d","ht.ntflux","ht.ntflux_u","ht.ntflux_d","cd.nIs","cd.nIl","cd.nIs_u","cd.nIs_d","reacf(phis)","reacf(T)","up(ht.dfluxy)","down(ht.dfluxy)"})try{m.result().numerical("lineprobe").set("expr",new String[]{e});System.out.println("BOUNDARY_PROBE|"+b+"|"+e+"|"+m.result().numerical("lineprobe").getReal()[0][0]);}catch(Throwable er){System.out.println("PROBE_ERROR|"+b+"|"+e);}
  }
  m.result().numerical().create("globalprobe","EvalGlobal");m.result().numerical("globalprobe").set("data","dset1");m.result().numerical("globalprobe").set("solnum",6);
  for(String e:new String[]{"ht.QInt","ht.heatBalance","ht.ntfluxInt","ht.dEiInt","cd.Is","cd.Il"})try{m.result().numerical("globalprobe").set("expr",new String[]{e});System.out.println("GLOBAL|"+e+"|"+m.result().numerical("globalprobe").getReal()[0][0]);}catch(Throwable er){System.out.println("GLOBAL_ERROR|"+e);}
  m.save(a[1],"java");ModelUtil.remove("Details");System.exit(0);
 }
}
