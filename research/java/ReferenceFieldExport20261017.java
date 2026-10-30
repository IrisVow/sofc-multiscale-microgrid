import com.comsol.model.*;import com.comsol.model.util.*;import java.io.*;import java.util.*;import java.util.zip.*;
/** Native mesh nodes, domain-qualified limits, full precision, no clipping/smoothing. */
public class ReferenceFieldExport20261017{
 static final double[] Y={0,.0008,.0013,.00132,.00182,.00262};
 static final String[] N={"T","u","v","p","w1","w2","w3","w4","phis","phil","rho","htQ","qav_an","qav_ca","CH2","CH2O","CO2","CN2","isx","isy","ilx","ily","iloc_pcar1","iloc_panr1","eta_pcar1","eta_panr1","Eeq_pcar1","Eeq_panr1","ilim_pcar1","ilim_panr1","rH2","rH2O","rO2","V_applied"};
 static String expr(int d,int k){
  if(k==0)return "T";
  if(k==8)return d==2||d==4?"phis":null;
  if(k==9)return d>=2&&d<=4?"phil":null;
  if(k==11)return "ht.Q";
  if(k==12)return d==4?"cd.iloc_panr1*Av_an":null;
  if(k==13)return d==2?"cd.iloc_pcar1*Av_ca":null;
  if(k==33)return "V_cell-V_step_amp*step_load(if(isdefined(t),t,0[s]))";
  if(k>=18&&k<=19)return d==2||d==4?"cd."+N[k]:null;
  if(k>=20&&k<=21)return d>=2&&d<=4?"cd."+N[k]:null;
  if(k>=22&&k<=29)return d==(k%2==0?2:4)?"cd."+N[k]:null;
  if(k>=30&&k<=32)return d!=3?"bt_R"+(k-29):null;
  if(d==3)return null;
  if(k>=14&&k<=17)return new String[]{"bt_H2*tcs2.rho/M_H2","bt_H2O*tcs2.rho/M_H2O","bt_O2*tcs2.rho/M_O2","bt_N2*tcs2.rho/M_N2"}[k-14];
  if(k>=4&&k<=7)return new String[]{"bt_H2","bt_H2O","bt_O2","bt_N2"}[k-4];
  if(k==10)return "tcs2.rho";
  return N[k];
 }
 public static void main(String[] a){try{
  ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(false);Model m=ModelUtil.load("ReferenceFieldExport",a[0]);
  String dataset=a[2],cid=a[3];String[] nums=a[4].split(","),times=a[5].split(",");if(nums.length!=times.length)throw new RuntimeException("Time/solution count mismatch");
  double[][] vertices=m.mesh("mesh1").getVertex();m.result().numerical().create("referenceNativeNodes","Interp");NumericalFeature f=m.result().numerical("referenceNativeNodes");f.set("data",dataset);f.set("ext",0);
  PrintWriter p=new PrintWriter(new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(a[1])),"UTF-8"));
  p.print("case_id,solnum,t_s,domain,x,y");for(String n:N)p.print(","+n);p.println();long rows=0;
  for(int ti=0;ti<nums.length;ti++)for(int d=1;d<=5;d++){
   ArrayList<Integer> ids=new ArrayList<>();for(int j=0;j<vertices[0].length;j++)if(vertices[1][j]>=Y[d-1]-1e-14&&vertices[1][j]<=Y[d]+1e-14)ids.add(j);
   double[][] xy=new double[2][ids.size()];for(int j=0;j<ids.size();j++){xy[0][j]=vertices[0][ids.get(j)];xy[1][j]=vertices[1][ids.get(j)];}
   f.selection().set(new int[]{d});f.set("coord",xy);f.set("solnum",Integer.parseInt(nums[ti]));
   ArrayList<String> es=new ArrayList<>();ArrayList<Integer> ks=new ArrayList<>();for(int k=0;k<N.length;k++)if(expr(d,k)!=null){es.add(expr(d,k));ks.add(k);}
   f.set("expr",es.toArray(new String[0]));double[][][] values=f.getData();double[][] cols=new double[N.length][];for(int j=0;j<ks.size();j++)cols[ks.get(j)]=values[j][0];
   for(int j=0;j<ids.size();j++){
    p.printf(Locale.US,"%s,%s,%s,%d,%.17g,%.17g",cid,nums[ti],times[ti],d,xy[0][j],xy[1][j]);
    for(int k=0;k<N.length;k++){if(cols[k]==null)p.print(",NaN");else{if(!Double.isFinite(cols[k][j]))throw new RuntimeException("Nonfinite active field "+N[k]+" domain="+d+" node="+j);p.printf(Locale.US,",%.17g",cols[k][j]);}}
    p.println();rows++;
   }p.flush();System.out.println("EXPORTED|"+cid+"|sol="+nums[ti]+"|domain="+d+"|rows="+ids.size());System.out.flush();
  }
  if(p.checkError())throw new IOException("Output error");p.close();System.out.println("FIELD_EXPORT_COMPLETE|rows="+rows);System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
