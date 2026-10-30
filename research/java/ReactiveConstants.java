import com.comsol.model.*;import com.comsol.model.util.*;import java.io.*;import java.util.*;
public class ReactiveConstants{
 public static void main(String[] a){try{ModelUtil.initStandalone(false);ModelUtil.showProgress(false);Model m=ModelUtil.load("Constants",a[0]);PrintWriter p=new PrintWriter(a[1],"UTF-8");p.println("name,value_SI");for(String s:new String[]{"F_const","R_const","M_H2","M_H2O","M_O2","M_N2","p_ref","H_ch","L","C_H2_in","C_O2_in"})p.printf(Locale.US,"%s,%.17g%n",s,m.param().evaluate(s));p.close();System.out.println("CONSTANTS_EXPORTED");System.exit(0);}catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
