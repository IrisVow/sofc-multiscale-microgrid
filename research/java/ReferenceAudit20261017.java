import com.comsol.model.*;
import com.comsol.model.util.*;
import com.comsol.model.physics.*;
import java.io.*;
import java.util.*;
public class ReferenceAudit20261017 {
 static PrintWriter out;
 static void props(com.comsol.model.PropFeature f,String path){
  out.println("NODE|"+path+"|"+f.getType());
  for(String k:f.properties())try{out.println(k+"="+Arrays.toString(f.getStringArray(k)));}catch(Throwable e){try{out.println(k+"="+f.getString(k));}catch(Throwable ignored){}}
 }
 static void physics(PhysicsFeature f,String path){
  out.println("NODE|"+path+"|"+f.getType());
  for(String k:f.properties())try{out.println(k+"="+Arrays.toString(f.getStringArray(k)));}catch(Throwable e){try{out.println(k+"="+f.getString(k));}catch(Throwable ignored){}}
  for(String table:new String[]{"Weak","Expression"})try{for(String[] r:f.featureInfo("info").getInfoTable(table))out.println(table+"|"+Arrays.toString(r));}catch(Throwable e){out.println("TABLE_UNAVAILABLE|"+table+"|"+e.getMessage());}
  for(String t:f.feature().tags())physics(f.feature(t),path+"/"+t);
 }
 static void solver(SolverFeature f,String path){props(f,path);for(String t:f.feature().tags())solver(f.feature(t),path+"/"+t);}
 public static void main(String[] a)throws Exception{
  ModelUtil.initStandalone(false);ModelUtil.showPlots(false);
  Model m=ModelUtil.load("ReferenceAudit",a[0]);out=new PrintWriter(a[1],"UTF-8");
  out.println("MODEL|"+a[0]);out.println("VERTICES|"+m.mesh("mesh1").getVertex()[0].length);
  for(String p:m.param().varnames())out.println("PARAM|"+p+"|"+m.param().get(p));
  for(String t:m.component("comp1").common().tags())props(m.component("comp1").common(t),"common/"+t);
  for(String p:new String[]{"fp","tbin","ht","siec"})for(String t:m.physics(p).feature().tags())physics(m.physics(p).feature(t),"physics/"+p+"/"+t);
  for(String s:m.study().tags())for(String t:m.study(s).feature().tags())props(m.study(s).feature(t),"study/"+s+"/"+t);
  for(String s:new String[]{"sol1","sol2"})for(String t:m.sol(s).feature().tags())solver(m.sol(s).feature(t),"solver/"+s+"/"+t);
  out.close();System.out.println("READ_ONLY_AUDIT_COMPLETE");System.exit(0);
 }
}
