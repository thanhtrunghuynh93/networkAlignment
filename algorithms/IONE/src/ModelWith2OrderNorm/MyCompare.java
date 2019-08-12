package ModelWith2OrderNorm;

import java.util.Comparator;


public class MyCompare implements Comparator   {  
	    public int compare(Object o1,Object o2) {  
	        String s1 = (String) o1;  
	        String s2 = (String) o2;  
	        
	        String[] array1=s1.split("\\s+");
	        String[] array2=s2.split("\\s+");
	        double value1=Double.parseDouble(array1[2]);
	        double value2=Double.parseDouble(array2[2]);
	        if(value1>value2)
	        {
	        	return 1;
	        }
	        else
	        {
	        	return -1;
	        }
	    }  
	}  


