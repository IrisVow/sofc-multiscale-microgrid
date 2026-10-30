import com.comsol.model.*;import com.comsol.model.util.*;import java.io.*;import java.util.*;
/** Native extrema, Faradaic currents, inlet flux, and the locked per-domain grid. */
public class ReferenceTransientMetrics20261018 {
 static Model m;static int sol;
 static double value(String tag,String expr,int[] ids){
  NumericalFeature f=m.result().numerical(tag);f.selection().set(ids);f.set("solnum",sol);f.set("expr",new String[]{expr});return f.getReal()[0][0];
 }
 public static void main(String[] a){try{
  ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(false);m=ModelUtil.load("TimeMetrics",a[0]);
  for(String type:new String[]{"IntSurface","IntLine","MaxSurface","MinSurface","Interp"}){m.result().numerical().create("tm"+type,type);m.result().numerical("tm"+type).set("data","refTimeData");}
  double[] times=m.sol("refTime").getPVals();String[] planned=a[2].split(",");
  if(times.length!=planned.length)throw new RuntimeException("Saved solution count differs from locked outputs");
  PrintWriter p=new PrintWriter(a[1]+"_metrics.csv","UTF-8"),g=new PrintWriter(a[1]+"_grid.csv","UTF-8");
  p.println("solnum,t_s,metric,value");g.println("solnum,t_s,domain,x,y,T");
  double[] Y={0,.0008,.0013,.00132,.00182,.00262};
  for(int j=0;j<times.length;j++){
   sol=j+1;double time=times[j];if(Math.abs(time-Double.parseDouble(planned[j]))>1e-8)throw new RuntimeException("Saved time mismatch");
   LinkedHashMap<String,Double> v=new LinkedHashMap<>();
   v.put("Tmax_K",value("tmMaxSurface","T",new int[]{1,2,3,4,5}));v.put("Tmin_K",value("tmMinSurface","T",new int[]{1,2,3,4,5}));
   v.put("max_temperature_change_from_initial_K",value("tmMaxSurface","abs(T-withsol('sol1',T))",new int[]{1,2,3,4,5}));
   v.put("max_binary_change_from_initial",value("tmMaxSurface","abs(bA-withsol('sol1',bA))",new int[]{1,2,4,5}));
   double ia=value("tmIntSurface","cd.iloc_panr1*Av_an",new int[]{4}),ic=-value("tmIntSurface","cd.iloc_pcar1*Av_ca",new int[]{2});
   double consume=-value("tmIntSurface","bt_R1",new int[]{4}),inlet=-value("tmIntLine","tcs2.ntflux_bA",new int[]{9});
   v.put("anode_current_A_m",ia);v.put("cathode_current_A_m",ic);v.put("H2_consumed_kg_m_s",consume);v.put("H2_inlet_kg_m_s",inlet);
   v.put("instantaneous_FU",consume/inlet);v.put("applied_voltage_V",value("tmMaxSurface","V_cell-V_step_amp*step_load(t)",new int[]{2}));
   v.put("max_abs_cathode_eta_V",value("tmMaxSurface","abs(cd.eta_pcar1)",new int[]{2}));
   v.put("max_abs_anode_eta_V",value("tmMaxSurface","abs(cd.eta_panr1)",new int[]{4}));
   for(String sp:new String[]{"bt_H2","bt_H2O","bt_O2","bt_N2"}){
    v.put("min_"+sp,value("tmMinSurface",sp,new int[]{1,2,4,5}));v.put("max_"+sp,value("tmMaxSurface",sp,new int[]{1,2,4,5}));
   }
   for(int dom=1;dom<=5;dom++){
    double[] xx=m.mesh("mesh1").getVertex()[0]; // Full domain-limited native nodes for Tmax.
    double[][] vertices=m.mesh("mesh1").getVertex();ArrayList<Integer> ids=new ArrayList<>();
    for(int k=0;k<xx.length;k++)if(vertices[1][k]>=Y[dom-1]-1e-14&&vertices[1][k]<=Y[dom]+1e-14)ids.add(k);
    double[][] nativeXY=new double[2][ids.size()];for(int k=0;k<ids.size();k++){nativeXY[0][k]=vertices[0][ids.get(k)];nativeXY[1][k]=vertices[1][ids.get(k)];}
    NumericalFeature f=m.result().numerical("tmInterp");f.selection().set(new int[]{dom});f.set("solnum",sol);f.set("coord",nativeXY);f.set("expr",new String[]{"T"});f.set("ext",0);
    double max=-Double.MAX_VALUE;for(double temp:f.getData()[0][0]){if(!Double.isFinite(temp))throw new RuntimeException("Nonfinite native temperature");max=Math.max(max,temp);}v.put("domain"+dom+"_native_Tmax_K",max);
    int nx=401,ny=9;double[][] xy=new double[2][nx*ny];int k=0;for(int ix=0;ix<nx;ix++)for(int iy=0;iy<ny;iy++){xy[0][k]=.1*(ix+.5)/nx;xy[1][k]=Y[dom-1]+(Y[dom]-Y[dom-1])*(iy+.5)/ny;k++;}
    f.set("coord",xy);double[] temps=f.getData()[0][0];for(k=0;k<temps.length;k++){if(!Double.isFinite(temps[k]))throw new RuntimeException("Nonfinite grid temperature");g.printf(Locale.US,"%d,%.17g,%d,%.17g,%.17g,%.17g%n",sol,time,dom,xy[0][k],xy[1][k],temps[k]);}
   }
   for(Map.Entry<String,Double> e:v.entrySet()){if(!Double.isFinite(e.getValue()))throw new RuntimeException("Nonfinite metric "+e.getKey());p.printf(Locale.US,"%d,%.17g,%s,%.17g%n",sol,time,e.getKey(),e.getValue());}
   p.flush();g.flush();System.out.println("METRICS_COMPLETE|sol="+sol+"|t="+time);System.out.flush();
  }
  if(p.checkError()||g.checkError())throw new IOException("Metric output failed");p.close();g.close();System.out.println("TIME_METRICS_COMPLETE");System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
