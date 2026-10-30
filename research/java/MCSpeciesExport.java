import com.comsol.model.*;
import com.comsol.model.util.*;
import com.comsol.model.physics.*;
import java.io.*;
import java.util.*;

public class MCSpeciesExport extends ReactiveExport {
 static boolean wall(int b){return b==3||b==6||b==7||b==8||b==13||b==15;}
 static String sidedDiff(String sp,int d){
  if(sp.equals("w4"))return "-("+sidedDiff("w1",d)+"+"+sidedDiff("w2",d)+"+"+sidedDiff("w3",d)+")";
  return "if(up(dom)=="+d+",-uflux_spatial("+sp+")/tcs.d,-dflux_spatial("+sp+")/tcs.d)";
 }
 static void eq(PhysicsFeature f,String path,PrintWriter p){
  for(String key:new String[]{"Expression","Weak","Constraint"}){
   p.println("FEATURE|"+path+"|"+key);
   try{for(String[] row:f.featureInfo("info").getInfoTable(key))p.println(Arrays.toString(row));}catch(Throwable e){p.println("ERROR|"+e.getMessage());}
  }
  p.flush();for(String tag:f.feature().tags())eq(f.feature(tag),path+"/"+tag,p);
 }
 public static void main(String[] a){try{audit(a);}catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
 static void audit(String[] a)throws Exception{
  ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(false);
  m=ModelUtil.load("MCSpeciesExport",a[0]);data="dset1";sol=1;cid=a[2];String prefix=a[1];
  if(Math.abs(m.param().evaluate("alpha_rxn")-1)>1e-12)throw new RuntimeException("Not a full-source state");
  m.result().numerical().create("clArea","IntSurface");m.result().numerical().create("clLine","IntLine");m.result().numerical().create("clProbe","Interp");
  summary(prefix);gasProfiles(prefix);dumpPoints(prefix+"_"+cid+"_points.csv");
  PrintWriter ep=new PrintWriter(prefix+"_recursive_equation_view.txt","UTF-8");
  for(String ph:new String[]{"fp","tcs"})for(String tag:m.physics(ph).feature().tags())eq(m.physics(ph).feature(tag),ph+"/"+tag,ep);ep.close();
  w=new PrintWriter(prefix+"_flux_integrals.csv","UTF-8");w.println("case,solnum,quadrature_order,kind,domain,boundary,metric,unit,value,status");
  for(int ord:new int[]{4,8}){order=ord;
   for(int d:new int[]{1,2,4,5}){
    record("volume",d,0,"mass_source","kg/(m*s)","fp.Qm");
    record("volume",d,0,"species_net_source","kg/(m*s)","tcs.Qmass");
    record("volume",d,0,"mass_divergence","kg/(m*s)","d(fp.rho*u,x)+d(fp.rho*v,y)");
    record("volume",d,0,"continuity_residual","kg/(m*s)","fp.continuityEquation");
    record("volume",d,0,"continuity_residual_abs","kg/(m*s)","abs(fp.continuityEquation)");
    for(String sp:new String[]{"w1","w2","w3","w4"}){
     record("volume",d,0,"source_"+sp,"kg/(m*s)","tcs.R_"+sp);
     record("volume",d,0,"weighted_continuity_"+sp,"kg/(m*s)",sp+"*fp.continuityEquation");
     record("volume",d,0,"NC_species_"+sp,"kg/(m*s)","fp.rho*(tcs.u*d("+sp+",x)+tcs.v*d("+sp+",y))+tcs.Qmass*"+sp+"-tcs.R_"+sp);
    }
   }
   for(int b=1;b<=m.geom("geom1").getNBoundaries();b++){
    double len=integ("IntLine",b,"1"),x=integ("IntLine",b,"x")/len,y=integ("IntLine",b,"y")/len;
    for(int d:new int[]{1,2,4,5}){
     double nx=0,ny=0;
     if(Math.abs(x)<1e-10&&y>Y[d-1]&&y<Y[d])nx=-1;
     else if(Math.abs(x-.1)<1e-10&&y>Y[d-1]&&y<Y[d])nx=1;
     else if(x>0&&x<.1&&Math.abs(y-Y[d-1])<1e-10)ny=-1;
     else if(x>0&&x<.1&&Math.abs(y-Y[d])<1e-10)ny=1;
     else continue;
     String vel="(u*"+nx+"+v*"+ny+")",flow=trace("fp.rho*"+vel,d);
     record("boundary",d,b,"mass_raw","kg/(m*s)",flow);
     record("boundary",d,b,"mass_raw_abs","kg/(m*s)","abs("+flow+")");
     if(wall(b)){
      record("boundary",d,b,"wall_native_convective","kg/(m*s)","fp.rhoFace*fp.un_here");
      record("boundary",d,b,"wall_prescribed_mass","kg/(m*s)","fp.rhoFace*fp.un_there");
      record("boundary",d,b,"wall_testp_coefficient","kg/(m*s)","fp.contCoeffFace*fp.unJump");
      record("boundary",d,b,"wall_weak_mass_flux","kg/(m*s)","fp.rhoFace*fp.un_here-fp.contCoeffFace*fp.unJump");
      record("boundary",d,b,"wall_normal_velocity_abs","m2/s","abs(fp.un_here)");
     }
     for(String sp:new String[]{"w1","w2","w3","w4"}){
      String conv=trace("fp.rho*"+sp+"*"+vel,d),diff=sidedDiff(sp,d),total="("+diff+")+ ("+conv+")";
      record("boundary",d,b,"convective_"+sp,"kg/(m*s)",conv);
      record("boundary",d,b,"diff_one_side_"+sp,"kg/(m*s)",diff);
      record("boundary",d,b,"total_one_side_"+sp,"kg/(m*s)",total);
      String numerical=sp.equals("w4")&&wall(b)?"("+total+")-mc_alpha*fp.contCoeffFace*fp.unJump":total;
      record("boundary",d,b,"numerical_total_"+sp,"kg/(m*s)",numerical);
      record("boundary",d,b,"numerical_total_abs_"+sp,"kg/(m*s)","abs("+numerical+")");
      String constitutive=trace("tcs.j_"+sp+"x*"+nx+"+tcs.j_"+sp+"y*"+ny,d)+"+("+conv+")";
      record("boundary",d,b,"constitutive_total_"+sp,"kg/(m*s)",constitutive);

      record("boundary",d,b,"total_averaged_builtin_"+sp,"kg/(m*s)",spTrace("tcs.ntflux_"+sp,d,b));
      if(wall(b))record("boundary",d,b,"wall_weighted_testp_"+sp,"kg/(m*s)",trace(sp,d)+"*fp.contCoeffFace*fp.unJump");
     }
    }
   }
  }
  w.close();ModelUtil.remove("MCSpeciesExport");System.out.println("FLUX_AUDIT_EXPORTED|"+cid);System.exit(0);
 }
}
