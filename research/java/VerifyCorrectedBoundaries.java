import com.comsol.model.*;import com.comsol.model.util.*;import com.comsol.model.physics.*;import java.util.*;import java.io.*;
public class VerifyCorrectedBoundaries {
 public static void main(String[] a)throws Exception{
  ModelUtil.initStandalone(false);Model m=ModelUtil.load("BoundaryAudit",a[0]);
  for(String ph:new String[]{"fp","ht"})for(String tag:m.physics(ph).feature().tags())if(tag.equals("out1")||tag.equals("ifl1")||tag.equals("hf1")){
   PhysicsFeature f=m.physics(ph).feature(tag);System.out.println("NODE|"+ph+"|"+tag);
   for(String prop:f.properties())try{System.out.println(prop+"="+f.getString(prop));}catch(Throwable e){}
  }
  for(String key:new String[]{"Compressibility","pref","IncludeGravity","UseReducedPressure"})try{System.out.println("FP_PHYSICAL|"+key+"="+m.physics("fp").prop("PhysicalModelProperty").getString(key));}catch(Throwable e){}
  m.result().numerical().create("boundaryCheck","Interp");m.result().numerical("boundaryCheck").set("data","dset2");
  String[] expr={"T","u","v","p","tcs.rho","d(T,x)","d(T,y)","d(u,x)","d(u,y)","d(v,x)","d(v,y)","t"};
  m.result().numerical("boundaryCheck").set("expr",expr);m.result().numerical("boundaryCheck").set("solnum",new int[]{Integer.parseInt(a[2])});
  ArrayList<double[]> xy=new ArrayList<>();ArrayList<String> names=new ArrayList<>();
  for(String dom:new String[]{"air","fuel"})for(String side:new String[]{"inlet","outlet"})for(int j=1;j<16;j++){
   double lo=dom.equals("air")?0:.00182,hi=dom.equals("air")?.0008:.00262;xy.add(new double[]{side.equals("inlet")?0:.1,lo+(hi-lo)*j/16.});names.add(dom+"_"+side);
  }
  for(String dom:new String[]{"air","fuel"})for(int j=1;j<16;j++){xy.add(new double[]{.1*j/16.,dom.equals("air")?0:.00262});names.add(dom+"_wall");}
  double[][] coords=new double[2][xy.size()];for(int j=0;j<xy.size();j++){coords[0][j]=xy.get(j)[0];coords[1][j]=xy.get(j)[1];}
  m.result().numerical("boundaryCheck").set("coord",coords);m.result().numerical("boundaryCheck").set("ext",0.);
  double[][][] v=m.result().numerical("boundaryCheck").getData();
  try(PrintWriter w=new PrintWriter(a[1],"UTF-8")){
   w.println("boundary,x,y,T,u,v,p,rho,Tx,Ty,ux,uy,vx,vy,t_s");
   for(int j=0;j<xy.size();j++){w.print(names.get(j));w.printf(Locale.US,",%.17g,%.17g",coords[0][j],coords[1][j]);for(int k=0;k<expr.length;k++)w.printf(Locale.US,",%.17g",v[k][0][j]);w.println();}
  }
  ModelUtil.remove("BoundaryAudit");System.out.println("NO_SOLVE_NO_SAVE");System.exit(0);
 }
}
