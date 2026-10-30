import com.comsol.model.*;import com.comsol.model.util.*;import java.util.*;import java.io.*;
public class LFInterfaceExport extends ClosureExport{
 static String old(String e){return "withsol('sol1',("+e+"))";}
 static String mass(int d){return "if(up(dom)=="+d+",-uflux_spatial(lfR),-dflux_spatial(lfR))";}
 static String species(String sp,int d,double nx,double ny){String conv=trace("fp.rho*"+sp+"*(u*"+nx+"+v*"+ny+")",d);return old("("+FluxAuditExport.sidedDiff(sp,d)+")+("+conv+")");}
 public static void main(String[] a){try{ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(false);m=ModelUtil.load("LFInterface",a[0]);data="lfData";sol=1;cid=a[2];
  for(String tag:new String[]{"clLine","clArea"})if(Arrays.asList(m.result().numerical().tags()).contains(tag))m.result().numerical().remove(tag);
  m.result().numerical().create("clLine","IntLine");m.result().numerical().create("clArea","IntSurface");
  w=new PrintWriter(a[1]+"_species.csv","UTF-8");w.println("case,solnum,quadrature_order,kind,domain,boundary,metric,unit,value,status");
  for(int ord:new int[]{4,8}){order=ord;
   for(int d:new int[]{1,2,4,5})for(String sp:new String[]{"w1","w2","w3","w4"})record("volume",d,0,"source_"+sp,"kg/(m*s)",old("tcs.R_"+sp));
   for(int b=1;b<=m.geom("geom1").getNBoundaries();b++){double len=integ("IntLine",b,"1"),xx=integ("IntLine",b,"x")/len,yy=integ("IntLine",b,"y")/len;
    for(int d:new int[]{1,2,4,5}){double nx=0,ny=0;if(Math.abs(xx)<1e-10&&yy>Y[d-1]&&yy<Y[d])nx=-1;else if(Math.abs(xx-.1)<1e-10&&yy>Y[d-1]&&yy<Y[d])nx=1;else if(xx>0&&xx<.1&&Math.abs(yy-Y[d-1])<1e-10)ny=-1;else if(xx>0&&xx<.1&&Math.abs(yy-Y[d])<1e-10)ny=1;else continue;
     String sum="("+species("w1",d,nx,ny)+")+("+species("w2",d,nx,ny)+")+("+species("w3",d,nx,ny)+")";
     for(String sp:new String[]{"w1","w2","w3","w4"}){String flux=sp.equals("w4")?"("+mass(d)+")-("+sum+")":species(sp,d,nx,ny);String previous=species(sp,d,nx,ny);if(sp.equals("w4")&&FluxAuditExport.wall(b))previous="("+previous+")-"+old("mc_alpha*fp.contCoeffFace*fp.unJump");
      record("boundary",d,b,"species_recovered_"+sp,"kg/(m*s)",flux);record("boundary",d,b,"species_recovered_abs_"+sp,"kg/(m*s)","abs("+flux+")");record("boundary",d,b,"species_previous_"+sp,"kg/(m*s)",previous);
     }
    }
   }
  }w.close();PrintWriter p=new PrintWriter(a[1]+"_interface_L1.csv","UTF-8");p.println("case,quadrature_order,boundary,quantity,signed_jump,L1_jump,L1_flux_scale");
  for(int ord:new int[]{4,8}){order=ord;for(int b:new int[]{4,10}){int da=b==4?1:4,db=b==4?2:5;
   for(String quantity:new String[]{"mass","w1","w2","w3","w4"}){String fa=mass(da),fb=mass(db);if(!quantity.equals("mass")){if(quantity.equals("w4")){fa="("+fa+")-("+species("w1",da,0,1)+")-("+species("w2",da,0,1)+")-("+species("w3",da,0,1)+")";fb="("+fb+")-("+species("w1",db,0,-1)+")-("+species("w2",db,0,-1)+")-("+species("w3",db,0,-1)+")";}else{fa=species(quantity,da,0,1);fb=species(quantity,db,0,-1);}}
    String jump="("+fa+")+("+fb+")";p.printf(Locale.US,"%s,%d,%d,%s,%.17g,%.17g,%.17g%n",cid,ord,b,quantity,integ("IntLine",b,jump),integ("IntLine",b,"abs("+jump+")"),integ("IntLine",b,"abs("+fa+")+abs("+fb+")"));p.flush();
   }
  }}p.close();System.out.println("LOCAL_INTERFACE_EXPORT_COMPLETE");System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
