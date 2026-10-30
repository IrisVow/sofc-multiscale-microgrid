public class BinaryPhysicalMap{
 static String fuel="(y>0.00131[m])";
 static String expr(String e){
  String[] names={"H2","H2O","O2","N2"};
  for(int i=1;i<=4;i++){
   e=e.replace("tcs.R_w"+i,"bt_R"+i);
   for(String f:new String[]{"ntflux","ndflux"}){String t=i==1?"if("+fuel+",tcs2."+f+"_bA,0[kg/(m^2*s)])":i==2?"if("+fuel+",tcs2."+f+"_bB,0[kg/(m^2*s)])":i==3?"if("+fuel+",0[kg/(m^2*s)],tcs2."+f+"_bA)":"if("+fuel+",0[kg/(m^2*s)],tcs2."+f+"_bB)";e=e.replace("tcs."+f+"_w"+i,t);}
   e=e.replaceAll("(?<![A-Za-z0-9_])w"+i+"(?![A-Za-z0-9_])","bt_"+names[i-1]);
  }
  return e.replace("tcs.rho","tcs2.rho").replace("tcs.pA","tcs2.pA").replace("tcs.u","tcs2.u").replace("tcs.v","tcs2.v");
 }
}
