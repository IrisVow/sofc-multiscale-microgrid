import com.comsol.model.*;import com.comsol.model.util.*;import java.io.*;import java.util.*;import java.security.*;import java.nio.*;
public class LFFrozenRecovery extends ClosureExport{
 static String hash(double[] u)throws Exception{MessageDigest h=MessageDigest.getInstance("SHA-256");ByteBuffer b=ByteBuffer.allocate(8);for(double x:u){b.clear();b.putDouble(x);h.update(b.array());}StringBuilder s=new StringBuilder();for(byte x:h.digest())s.append(String.format("%02x",x));return s.toString();}
 public static void main(String[] a){try{
  ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(a[3]);m=ModelUtil.load("LFFrozen",a[0]);String before=hash(m.sol("sol1").getU());String aux="";
  m.study("lfStudy").feature("stat").set("useadvanceddisable","on");m.study("lfStudy").feature("stat").set("disabledcommon",new String[]{"comp1/mcVolume","comp1/mcWall"});
  ArrayList<String> primary=new ArrayList<>();for(String tag:m.physics().tags()){primary.add(tag);primary.add(tag.equals("lfMass")?"off":"on");}m.study("std1").feature("stat").set("activate",primary.toArray(new String[0]));
  for(String tag:m.sol().tags())if(m.sol(tag).study().equals("lfStudy"))aux=tag;if(aux.isEmpty())throw new RuntimeException("Auxiliary solver missing");
  m.sol(aux).feature("s1").set("stol","1e-10");System.out.println("FROZEN_RECOVERY_START|"+aux);System.out.flush();m.sol(aux).runAll();
  String after=hash(m.sol("sol1").getU());if(!before.equals(after))throw new RuntimeException("Frozen source solution changed");
  m.result().dataset().create("lfData","Solution");m.result().dataset("lfData").set("solution",aux);data="lfData";sol=1;cid=a[2];
  for(String tag:new String[]{"clLine","clArea"})if(Arrays.asList(m.result().numerical().tags()).contains(tag))m.result().numerical().remove(tag);
  m.result().numerical().create("clLine","IntLine");m.result().numerical().create("clArea","IntSurface");w=new PrintWriter(a[1]+"_integrals.csv","UTF-8");w.println("case,solnum,quadrature_order,kind,domain,boundary,metric,unit,value,status");
  for(int ord:new int[]{4,8}){order=ord;
   for(int d:new int[]{1,2,4,5}){record("volume",d,0,"mass_source","kg/(m*s)","withsol('sol1',fp.Qm)");record("volume",d,0,"source_abs","kg/(m*s)","abs(withsol('sol1',fp.Qm))");record("volume",d,0,"residual_lifting","kg/(m*s)","lfR/1[s]");record("volume",d,0,"residual_lifting_abs","kg/(m*s)","abs(lfR/1[s])");record("volume",d,0,"area","m2","1");}
   for(int b=1;b<=m.geom("geom1").getNBoundaries();b++){
    double len=integ("IntLine",b,"1"),xx=integ("IntLine",b,"x")/len,yy=integ("IntLine",b,"y")/len;
    for(int d:new int[]{1,2,4,5}){double nx=0,ny=0;if(Math.abs(xx)<1e-10&&yy>Y[d-1]&&yy<Y[d])nx=-1;else if(Math.abs(xx-.1)<1e-10&&yy>Y[d-1]&&yy<Y[d])nx=1;else if(xx>0&&xx<.1&&Math.abs(yy-Y[d-1])<1e-10)ny=-1;else if(xx>0&&xx<.1&&Math.abs(yy-Y[d])<1e-10)ny=1;else continue;
     String raw=trace("withsol('sol1',fp.rho*(u*"+nx+"+v*"+ny+"))",d),rec="if(up(dom)=="+d+",-uflux_spatial(lfR),-dflux_spatial(lfR))";
     record("boundary",d,b,"mass_raw","kg/(m*s)",raw);record("boundary",d,b,"mass_recovered","kg/(m*s)",rec);record("boundary",d,b,"mass_recovered_abs","kg/(m*s)","abs("+rec+")");record("boundary",d,b,"recovery_change_abs","kg/(m*s)","abs(("+rec+")-("+raw+"))");
    }
   }
  }w.close();
  PrintWriter meta=new PrintWriter(a[1]+"_metadata.json","UTF-8");meta.println("{\"source_solution_unchanged\":true,\"source_solution_sha256\":\""+before+"\",\"auxiliary_solver\":\""+aux+"\",\"source_budget_not_hidden\":true}");meta.close();
  m.label("Frozen-field conservative mass-flux recovery; source residual must be validated");m.save(a[1]+".mph");m.save(a[1]+"_history.java","java");System.out.println("FROZEN_RECOVERY_COMPLETE");System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
