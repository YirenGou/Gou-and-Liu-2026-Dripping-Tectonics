	/* Insert the following part in function double viscalc(double mtk, double mpb, double x, double y, long int mm1, int mm2) */

		/*Flow laws with Dislocation and Grain Boundary Sliding*/
		if (mm2 == 7 || mm2 == 8|| mm2 == 14)
		{
			double dia, pvalue, cw, rvalue;
			dia = 60000;
			if(marky[mm2]<180000) dia = (60000-300)/(180000-75000)*(marky[mm2]-75000)+300;		/*Grain size microns*/
			if(markx[mm2]<500000) cw = 760-markx[mm2]/500000*750;			/*Water content ppm*/
			if(markx[mm2]>=500000) cw = 10;		
			/*Dislocation*/
			rvalue = 1.93;		/*Water content exponent*/
			sduct = pow(marknu[mm2]*e1*epsin*pow(cw,-rvalue),1.0/markmm[mm2]); /*(A^-1 * epsin * e1 * d^p * Cw^-r)^1/n*/
			/* Inverted value of power-low NU calc */
			if(epsin>0) nupowl=1.0/(0.5*sduct/epsin);		
			
			/*Grain Boundary Sliding*/
			pvalue = 1.99;		/*Grain size exponent*/
			rvalue = -0.34;		/*Water content exponent*/
			/* T-P exponent for effective NU calc */
			e1=(653000+0.706*mpb)/rt; /*1 cm³/mol = 0.1 J/(mol⋅bar)*/
			if(e1>150.0) e1=150.0;
			e1=exp(e1);			
			sduct = pow(1.95e+22*e1*epsin*pow(dia,pvalue)*pow(cw,-rvalue),1.0/5.11); /*(A^-1 * epsin * e1 * d^p * Cw^-r)^1/n*/
			/* Inverted value of power-low NU calc */
			if(epsin>0) nupowl=nupowl+1.0/(0.5*sduct/epsin);	
		}