/* Generated structural sensitivity transport. No approximate sparsity. */
#define NX 24
#define NU 4
#define ND 33
#define NV 411
#define OUT (NX + NX*NX + 2*NX*NU + 2*NX)
static void initialize(double *v, const double *x0) {
    memset(v,0,NV*sizeof(double)); memcpy(v,x0,NX*sizeof(double));
    v[115]=1.0;
    v[128]=1.0;
    v[151]=1.0;
    v[162]=1.0;
    v[173]=1.0;
    v[315]=1.0;
    v[360]=1.0;
}
static int rhs(const double *restrict v, const double *u0, const double *u1,
               double scale, double beta, const double *parameters, double *restrict d) {
    double u[NU], f[NX], jac[QUAD_NNZ];
    for(int i=0;i<NU;++i) u[i]=u0[i]+beta*(u1[i]-u0[i]);
    quad_field_prepared(v,u,parameters,f,jac);
    memset(d,0,NV*sizeof(double));
    for(int i=0;i<NX;++i) {if(!isfinite(f[i])) return 2; d[i]=scale*f[i];}
    for(int i=0;i<QUAD_NNZ;++i) if(!isfinite(jac[i])) return 3;
    d[39]=f[0];
    d[55]=f[1];
    d[70]=f[2];
    d[85]=f[3];
    d[100]=f[4];
    d[114]=f[5];
    d[126]=f[6];
    d[138]=f[7];
    d[150]=f[8];
    d[160]=f[9];
    d[170]=f[10];
    d[180]=f[11];
    d[200]=f[12];
    d[220]=f[13];
    d[240]=f[14];
    d[260]=f[15];
    d[280]=f[16];
    d[300]=f[17];
    d[301]=f[18];
    d[324]=f[19];
    d[346]=f[20];
    d[369]=f[21];
    d[391]=f[22];
    d[410]=f[23];
    { const double a=scale*jac[0];
      d[24]+=a;
      for(int j=0;j<15;++j) d[25+j]+=a*v[71+j];
    }
    { const double a=scale*jac[1];
      d[40]+=a;
      for(int j=0;j<15;++j) d[41+j]+=a*v[86+j];
    }
    { const double a=scale*jac[2];
      d[56]+=a;
      for(int j=0;j<14;++j) d[57+j]+=a*v[101+j];
    }
    { const double a=scale*jac[3];
      for(int j=0;j<2;++j) d[71+j]+=a*v[115+j];
      for(int j=0;j<6;++j) d[74+j]+=a*v[117+j];
      for(int j=0;j<3;++j) d[81+j]+=a*v[123+j];
      d[85]+=a*v[126];
    }
    { const double a=scale*jac[4];
      for(int j=0;j<2;++j) d[71+j]+=a*v[127+j];
      for(int j=0;j<6;++j) d[74+j]+=a*v[129+j];
      for(int j=0;j<3;++j) d[81+j]+=a*v[135+j];
      d[85]+=a*v[138];
    }
    { const double a=scale*jac[5];
      d[73]+=a;
      for(int j=0;j<2;++j) d[71+j]+=a*v[139+j];
      for(int j=0;j<6;++j) d[74+j]+=a*v[141+j];
      for(int j=0;j<3;++j) d[81+j]+=a*v[147+j];
      d[85]+=a*v[150];
    }
    { const double a=scale*jac[6];
      d[80]+=a*(1.0-beta);
      d[84]+=a*beta;
    }
    { const double a=scale*jac[7];
      for(int j=0;j<2;++j) d[86+j]+=a*v[115+j];
      for(int j=0;j<6;++j) d[89+j]+=a*v[117+j];
      for(int j=0;j<3;++j) d[96+j]+=a*v[123+j];
      d[100]+=a*v[126];
    }
    { const double a=scale*jac[8];
      for(int j=0;j<2;++j) d[86+j]+=a*v[127+j];
      for(int j=0;j<6;++j) d[89+j]+=a*v[129+j];
      for(int j=0;j<3;++j) d[96+j]+=a*v[135+j];
      d[100]+=a*v[138];
    }
    { const double a=scale*jac[9];
      d[88]+=a;
      for(int j=0;j<2;++j) d[86+j]+=a*v[139+j];
      for(int j=0;j<6;++j) d[89+j]+=a*v[141+j];
      for(int j=0;j<3;++j) d[96+j]+=a*v[147+j];
      d[100]+=a*v[150];
    }
    { const double a=scale*jac[10];
      d[95]+=a*(1.0-beta);
      d[99]+=a*beta;
    }
    { const double a=scale*jac[11];
      for(int j=0;j<8;++j) d[101+j]+=a*v[115+j];
      for(int j=0;j<3;++j) d[110+j]+=a*v[123+j];
      d[114]+=a*v[126];
    }
    { const double a=scale*jac[12];
      for(int j=0;j<8;++j) d[101+j]+=a*v[127+j];
      for(int j=0;j<3;++j) d[110+j]+=a*v[135+j];
      d[114]+=a*v[138];
    }
    { const double a=scale*jac[13];
      d[109]+=a*(1.0-beta);
      d[113]+=a*beta;
    }
    { const double a=scale*jac[14];
      for(int j=0;j<12;++j) d[115+j]+=a*v[115+j];
    }
    { const double a=scale*jac[15];
      for(int j=0;j<12;++j) d[115+j]+=a*v[127+j];
    }
    { const double a=scale*jac[16];
      for(int j=0;j<10;++j) d[117+j]+=a*v[151+j];
    }
    { const double a=scale*jac[17];
      for(int j=0;j<10;++j) d[117+j]+=a*v[161+j];
    }
    { const double a=scale*jac[18];
      for(int j=0;j<10;++j) d[117+j]+=a*v[171+j];
    }
    { const double a=scale*jac[19];
      for(int j=0;j<12;++j) d[127+j]+=a*v[115+j];
    }
    { const double a=scale*jac[20];
      for(int j=0;j<10;++j) d[129+j]+=a*v[161+j];
    }
    { const double a=scale*jac[21];
      for(int j=0;j<10;++j) d[129+j]+=a*v[171+j];
    }
    { const double a=scale*jac[22];
      for(int j=0;j<12;++j) d[139+j]+=a*v[115+j];
    }
    { const double a=scale*jac[23];
      for(int j=0;j<12;++j) d[139+j]+=a*v[127+j];
    }
    { const double a=scale*jac[24];
      for(int j=0;j<10;++j) d[141+j]+=a*v[161+j];
    }
    { const double a=scale*jac[25];
      for(int j=0;j<10;++j) d[141+j]+=a*v[171+j];
    }
    { const double a=scale*jac[26];
      for(int j=0;j<10;++j) d[151+j]+=a*v[161+j];
    }
    { const double a=scale*jac[27];
      for(int j=0;j<10;++j) d[151+j]+=a*v[171+j];
    }
    { const double a=scale*jac[28];
      d[154]+=a*(1.0-beta);
      d[157]+=a*beta;
    }
    { const double a=scale*jac[29];
      for(int j=0;j<10;++j) d[161+j]+=a*v[151+j];
    }
    { const double a=scale*jac[30];
      for(int j=0;j<10;++j) d[161+j]+=a*v[171+j];
    }
    { const double a=scale*jac[31];
      d[165]+=a*(1.0-beta);
      d[168]+=a*beta;
    }
    { const double a=scale*jac[32];
      for(int j=0;j<10;++j) d[171+j]+=a*v[151+j];
    }
    { const double a=scale*jac[33];
      for(int j=0;j<10;++j) d[171+j]+=a*v[161+j];
    }
    { const double a=scale*jac[34];
      d[176]+=a*(1.0-beta);
      d[179]+=a*beta;
    }
    { const double a=scale*jac[35];
      d[181]+=a;
      d[183]+=a*v[24];
      for(int j=0;j<6;++j) d[185+j]+=a*v[25+j];
      for(int j=0;j<9;++j) d[192+j]+=a*v[31+j];
    }
    { const double a=scale*jac[36];
      d[182]+=a;
      for(int j=0;j<7;++j) d[184+j]+=a*v[40+j];
      for(int j=0;j<9;++j) d[192+j]+=a*v[47+j];
    }
    { const double a=scale*jac[37];
      d[191]+=a;
      d[200]+=a*v[301];
    }
    { const double a=scale*jac[38];
      d[201]+=a;
      d[203]+=a*v[24];
      for(int j=0;j<6;++j) d[205+j]+=a*v[25+j];
      for(int j=0;j<9;++j) d[212+j]+=a*v[31+j];
    }
    { const double a=scale*jac[39];
      d[202]+=a;
      for(int j=0;j<7;++j) d[204+j]+=a*v[40+j];
      for(int j=0;j<9;++j) d[212+j]+=a*v[47+j];
    }
    { const double a=scale*jac[40];
      d[211]+=a;
      d[220]+=a*v[301];
    }
    { const double a=scale*jac[41];
      d[221]+=a;
      d[223]+=a*v[24];
      for(int j=0;j<6;++j) d[225+j]+=a*v[25+j];
      for(int j=0;j<9;++j) d[232+j]+=a*v[31+j];
    }
    { const double a=scale*jac[42];
      d[222]+=a;
      for(int j=0;j<7;++j) d[224+j]+=a*v[40+j];
      for(int j=0;j<9;++j) d[232+j]+=a*v[47+j];
    }
    { const double a=scale*jac[43];
      d[231]+=a;
      d[240]+=a*v[301];
    }
    { const double a=scale*jac[44];
      d[241]+=a;
      d[243]+=a*v[24];
      for(int j=0;j<6;++j) d[245+j]+=a*v[25+j];
      for(int j=0;j<9;++j) d[252+j]+=a*v[31+j];
    }
    { const double a=scale*jac[45];
      d[242]+=a;
      for(int j=0;j<7;++j) d[244+j]+=a*v[40+j];
      for(int j=0;j<9;++j) d[252+j]+=a*v[47+j];
    }
    { const double a=scale*jac[46];
      d[251]+=a;
      d[260]+=a*v[301];
    }
    { const double a=scale*jac[47];
      d[261]+=a;
      d[263]+=a*v[24];
      for(int j=0;j<6;++j) d[265+j]+=a*v[25+j];
      for(int j=0;j<9;++j) d[272+j]+=a*v[31+j];
    }
    { const double a=scale*jac[48];
      d[262]+=a;
      for(int j=0;j<7;++j) d[264+j]+=a*v[40+j];
      for(int j=0;j<9;++j) d[272+j]+=a*v[47+j];
    }
    { const double a=scale*jac[49];
      d[271]+=a;
      d[280]+=a*v[301];
    }
    { const double a=scale*jac[50];
      d[281]+=a;
      d[283]+=a*v[24];
      for(int j=0;j<6;++j) d[285+j]+=a*v[25+j];
      for(int j=0;j<9;++j) d[292+j]+=a*v[31+j];
    }
    { const double a=scale*jac[51];
      d[282]+=a;
      for(int j=0;j<7;++j) d[284+j]+=a*v[40+j];
      for(int j=0;j<9;++j) d[292+j]+=a*v[47+j];
    }
    { const double a=scale*jac[52];
      d[291]+=a;
      d[300]+=a*v[301];
    }
    { const double a=scale*jac[53];
      d[302]+=a;
      d[305]+=a*v[24];
      for(int j=0;j<6;++j) d[308+j]+=a*v[25+j];
      for(int j=0;j<9;++j) d[316+j]+=a*v[31+j];
    }
    { const double a=scale*jac[54];
      d[303]+=a;
      d[306]+=a*v[40];
      for(int j=0;j<6;++j) d[308+j]+=a*v[41+j];
      for(int j=0;j<9;++j) d[316+j]+=a*v[47+j];
    }
    { const double a=scale*jac[55];
      d[304]+=a;
      for(int j=0;j<3;++j) d[307+j]+=a*v[56+j];
      for(int j=0;j<3;++j) d[311+j]+=a*v[59+j];
      for(int j=0;j<9;++j) d[316+j]+=a*v[62+j];
    }
    { const double a=scale*jac[56];
      d[314]+=a;
      d[324]+=a*v[301];
    }
    { const double a=scale*jac[57];
      for(int j=0;j<23;++j) d[302+j]+=a*v[302+j];
    }
    { const double a=scale*jac[58];
      d[325]+=a;
      d[328]+=a*v[24];
      for(int j=0;j<6;++j) d[331+j]+=a*v[25+j];
      for(int j=0;j<9;++j) d[338+j]+=a*v[31+j];
    }
    { const double a=scale*jac[59];
      d[326]+=a;
      d[329]+=a*v[40];
      for(int j=0;j<6;++j) d[331+j]+=a*v[41+j];
      for(int j=0;j<9;++j) d[338+j]+=a*v[47+j];
    }
    { const double a=scale*jac[60];
      d[327]+=a;
      for(int j=0;j<3;++j) d[330+j]+=a*v[56+j];
      for(int j=0;j<3;++j) d[334+j]+=a*v[59+j];
      for(int j=0;j<9;++j) d[338+j]+=a*v[62+j];
    }
    { const double a=scale*jac[61];
      d[337]+=a;
      d[346]+=a*v[301];
    }
    { const double a=scale*jac[62];
      d[347]+=a;
      d[350]+=a*v[24];
      for(int j=0;j<6;++j) d[353+j]+=a*v[25+j];
      for(int j=0;j<9;++j) d[361+j]+=a*v[31+j];
    }
    { const double a=scale*jac[63];
      d[348]+=a;
      d[351]+=a*v[40];
      for(int j=0;j<6;++j) d[353+j]+=a*v[41+j];
      for(int j=0;j<9;++j) d[361+j]+=a*v[47+j];
    }
    { const double a=scale*jac[64];
      d[349]+=a;
      for(int j=0;j<3;++j) d[352+j]+=a*v[56+j];
      for(int j=0;j<3;++j) d[356+j]+=a*v[59+j];
      for(int j=0;j<9;++j) d[361+j]+=a*v[62+j];
    }
    { const double a=scale*jac[65];
      d[359]+=a;
      d[369]+=a*v[301];
    }
    { const double a=scale*jac[66];
      for(int j=0;j<23;++j) d[347+j]+=a*v[347+j];
    }
    { const double a=scale*jac[67];
      d[370]+=a;
      d[373]+=a*v[24];
      for(int j=0;j<6;++j) d[376+j]+=a*v[25+j];
      for(int j=0;j<9;++j) d[383+j]+=a*v[31+j];
    }
    { const double a=scale*jac[68];
      d[371]+=a;
      d[374]+=a*v[40];
      for(int j=0;j<6;++j) d[376+j]+=a*v[41+j];
      for(int j=0;j<9;++j) d[383+j]+=a*v[47+j];
    }
    { const double a=scale*jac[69];
      d[372]+=a;
      for(int j=0;j<3;++j) d[375+j]+=a*v[56+j];
      for(int j=0;j<3;++j) d[379+j]+=a*v[59+j];
      for(int j=0;j<9;++j) d[383+j]+=a*v[62+j];
    }
    { const double a=scale*jac[70];
      d[382]+=a;
      d[391]+=a*v[301];
    }
    { const double a=scale*jac[71];
      d[392]+=a;
      for(int j=0;j<3;++j) d[395+j]+=a*v[56+j];
      for(int j=0;j<12;++j) d[399+j]+=a*v[59+j];
    }
    { const double a=scale*jac[72];
      d[393]+=a;
      for(int j=0;j<15;++j) d[396+j]+=a*v[71+j];
    }
    { const double a=scale*jac[73];
      d[394]+=a;
      for(int j=0;j<15;++j) d[396+j]+=a*v[86+j];
    }
    { const double a=scale*jac[74];
      d[395]+=a;
      for(int j=0;j<2;++j) d[396+j]+=a*v[101+j];
      for(int j=0;j<12;++j) d[399+j]+=a*v[103+j];
    }
    { const double a=scale*jac[75];
      for(int j=0;j<2;++j) d[396+j]+=a*v[115+j];
      for(int j=0;j<6;++j) d[399+j]+=a*v[117+j];
      for(int j=0;j<3;++j) d[406+j]+=a*v[123+j];
      d[410]+=a*v[126];
    }
    { const double a=scale*jac[76];
      for(int j=0;j<2;++j) d[396+j]+=a*v[127+j];
      for(int j=0;j<6;++j) d[399+j]+=a*v[129+j];
      for(int j=0;j<3;++j) d[406+j]+=a*v[135+j];
      d[410]+=a*v[138];
    }
    { const double a=scale*jac[77];
      for(int j=0;j<6;++j) d[399+j]+=a*v[151+j];
      for(int j=0;j<3;++j) d[406+j]+=a*v[157+j];
      d[410]+=a*v[160];
    }
    { const double a=scale*jac[78];
      for(int j=0;j<6;++j) d[399+j]+=a*v[161+j];
      for(int j=0;j<3;++j) d[406+j]+=a*v[167+j];
      d[410]+=a*v[170];
    }
    { const double a=scale*jac[79];
      for(int j=0;j<6;++j) d[399+j]+=a*v[171+j];
      for(int j=0;j<3;++j) d[406+j]+=a*v[177+j];
      d[410]+=a*v[180];
    }
    return 0;
}
static void unpack(const double *v,const double *x0,const double *u0,const double *u1,
                   double scale,int free_time,double *out) {
    memset(out,0,OUT*sizeof(double));memcpy(out,v,NX*sizeof(double));
    out[24]=1.0;
    out[27]=v[24];
    out[30]=v[25];
    out[31]=v[26];
    out[32]=v[27];
    out[33]=v[28];
    out[34]=v[29];
    out[35]=v[30];
    out[600]=v[31];
    out[601]=v[32];
    out[602]=v[33];
    out[603]=v[34];
    out[696]=v[35];
    out[697]=v[36];
    out[698]=v[37];
    out[699]=v[38];
    out[792]=v[39];
    out[49]=1.0;
    out[52]=v[40];
    out[54]=v[41];
    out[55]=v[42];
    out[56]=v[43];
    out[57]=v[44];
    out[58]=v[45];
    out[59]=v[46];
    out[604]=v[47];
    out[605]=v[48];
    out[606]=v[49];
    out[607]=v[50];
    out[700]=v[51];
    out[701]=v[52];
    out[702]=v[53];
    out[703]=v[54];
    out[793]=v[55];
    out[74]=1.0;
    out[77]=v[56];
    out[78]=v[57];
    out[79]=v[58];
    out[81]=v[59];
    out[82]=v[60];
    out[83]=v[61];
    out[608]=v[62];
    out[609]=v[63];
    out[610]=v[64];
    out[611]=v[65];
    out[704]=v[66];
    out[705]=v[67];
    out[706]=v[68];
    out[707]=v[69];
    out[794]=v[70];
    out[99]=1.0;
    out[102]=v[71];
    out[103]=v[72];
    out[104]=v[73];
    out[105]=v[74];
    out[106]=v[75];
    out[107]=v[76];
    out[612]=v[77];
    out[613]=v[78];
    out[614]=v[79];
    out[615]=v[80];
    out[708]=v[81];
    out[709]=v[82];
    out[710]=v[83];
    out[711]=v[84];
    out[795]=v[85];
    out[124]=1.0;
    out[126]=v[86];
    out[127]=v[87];
    out[128]=v[88];
    out[129]=v[89];
    out[130]=v[90];
    out[131]=v[91];
    out[616]=v[92];
    out[617]=v[93];
    out[618]=v[94];
    out[619]=v[95];
    out[712]=v[96];
    out[713]=v[97];
    out[714]=v[98];
    out[715]=v[99];
    out[796]=v[100];
    out[149]=1.0;
    out[150]=v[101];
    out[151]=v[102];
    out[153]=v[103];
    out[154]=v[104];
    out[155]=v[105];
    out[620]=v[106];
    out[621]=v[107];
    out[622]=v[108];
    out[623]=v[109];
    out[716]=v[110];
    out[717]=v[111];
    out[718]=v[112];
    out[719]=v[113];
    out[797]=v[114];
    out[174]=v[115];
    out[175]=v[116];
    out[177]=v[117];
    out[178]=v[118];
    out[179]=v[119];
    out[624]=v[120];
    out[625]=v[121];
    out[626]=v[122];
    out[720]=v[123];
    out[721]=v[124];
    out[722]=v[125];
    out[798]=v[126];
    out[198]=v[127];
    out[199]=v[128];
    out[201]=v[129];
    out[202]=v[130];
    out[203]=v[131];
    out[628]=v[132];
    out[629]=v[133];
    out[630]=v[134];
    out[724]=v[135];
    out[725]=v[136];
    out[726]=v[137];
    out[799]=v[138];
    out[222]=v[139];
    out[223]=v[140];
    out[224]=1.0;
    out[225]=v[141];
    out[226]=v[142];
    out[227]=v[143];
    out[632]=v[144];
    out[633]=v[145];
    out[634]=v[146];
    out[728]=v[147];
    out[729]=v[148];
    out[730]=v[149];
    out[800]=v[150];
    out[249]=v[151];
    out[250]=v[152];
    out[251]=v[153];
    out[636]=v[154];
    out[637]=v[155];
    out[638]=v[156];
    out[732]=v[157];
    out[733]=v[158];
    out[734]=v[159];
    out[801]=v[160];
    out[273]=v[161];
    out[274]=v[162];
    out[275]=v[163];
    out[640]=v[164];
    out[641]=v[165];
    out[642]=v[166];
    out[736]=v[167];
    out[737]=v[168];
    out[738]=v[169];
    out[802]=v[170];
    out[297]=v[171];
    out[298]=v[172];
    out[299]=v[173];
    out[644]=v[174];
    out[645]=v[175];
    out[646]=v[176];
    out[740]=v[177];
    out[741]=v[178];
    out[742]=v[179];
    out[803]=v[180];
    out[312]=v[181];
    out[313]=v[182];
    out[315]=v[183];
    out[316]=v[184];
    out[318]=v[185];
    out[319]=v[186];
    out[320]=v[187];
    out[321]=v[188];
    out[322]=v[189];
    out[323]=v[190];
    out[324]=1.0;
    out[330]=v[191];
    out[648]=v[192];
    out[649]=v[193];
    out[650]=v[194];
    out[651]=v[195];
    out[744]=v[196];
    out[745]=v[197];
    out[746]=v[198];
    out[747]=v[199];
    out[804]=v[200];
    out[336]=v[201];
    out[337]=v[202];
    out[339]=v[203];
    out[340]=v[204];
    out[342]=v[205];
    out[343]=v[206];
    out[344]=v[207];
    out[345]=v[208];
    out[346]=v[209];
    out[347]=v[210];
    out[349]=1.0;
    out[354]=v[211];
    out[652]=v[212];
    out[653]=v[213];
    out[654]=v[214];
    out[655]=v[215];
    out[748]=v[216];
    out[749]=v[217];
    out[750]=v[218];
    out[751]=v[219];
    out[805]=v[220];
    out[360]=v[221];
    out[361]=v[222];
    out[363]=v[223];
    out[364]=v[224];
    out[366]=v[225];
    out[367]=v[226];
    out[368]=v[227];
    out[369]=v[228];
    out[370]=v[229];
    out[371]=v[230];
    out[374]=1.0;
    out[378]=v[231];
    out[656]=v[232];
    out[657]=v[233];
    out[658]=v[234];
    out[659]=v[235];
    out[752]=v[236];
    out[753]=v[237];
    out[754]=v[238];
    out[755]=v[239];
    out[806]=v[240];
    out[384]=v[241];
    out[385]=v[242];
    out[387]=v[243];
    out[388]=v[244];
    out[390]=v[245];
    out[391]=v[246];
    out[392]=v[247];
    out[393]=v[248];
    out[394]=v[249];
    out[395]=v[250];
    out[399]=1.0;
    out[402]=v[251];
    out[660]=v[252];
    out[661]=v[253];
    out[662]=v[254];
    out[663]=v[255];
    out[756]=v[256];
    out[757]=v[257];
    out[758]=v[258];
    out[759]=v[259];
    out[807]=v[260];
    out[408]=v[261];
    out[409]=v[262];
    out[411]=v[263];
    out[412]=v[264];
    out[414]=v[265];
    out[415]=v[266];
    out[416]=v[267];
    out[417]=v[268];
    out[418]=v[269];
    out[419]=v[270];
    out[424]=1.0;
    out[426]=v[271];
    out[664]=v[272];
    out[665]=v[273];
    out[666]=v[274];
    out[667]=v[275];
    out[760]=v[276];
    out[761]=v[277];
    out[762]=v[278];
    out[763]=v[279];
    out[808]=v[280];
    out[432]=v[281];
    out[433]=v[282];
    out[435]=v[283];
    out[436]=v[284];
    out[438]=v[285];
    out[439]=v[286];
    out[440]=v[287];
    out[441]=v[288];
    out[442]=v[289];
    out[443]=v[290];
    out[449]=1.0;
    out[450]=v[291];
    out[668]=v[292];
    out[669]=v[293];
    out[670]=v[294];
    out[671]=v[295];
    out[764]=v[296];
    out[765]=v[297];
    out[766]=v[298];
    out[767]=v[299];
    out[809]=v[300];
    out[474]=1.0;
    out[810]=v[301];
    out[480]=v[302];
    out[481]=v[303];
    out[482]=v[304];
    out[483]=v[305];
    out[484]=v[306];
    out[485]=v[307];
    out[486]=v[308];
    out[487]=v[309];
    out[488]=v[310];
    out[489]=v[311];
    out[490]=v[312];
    out[491]=v[313];
    out[498]=v[314];
    out[499]=v[315];
    out[676]=v[316];
    out[677]=v[317];
    out[678]=v[318];
    out[679]=v[319];
    out[772]=v[320];
    out[773]=v[321];
    out[774]=v[322];
    out[775]=v[323];
    out[811]=v[324];
    out[504]=v[325];
    out[505]=v[326];
    out[506]=v[327];
    out[507]=v[328];
    out[508]=v[329];
    out[509]=v[330];
    out[510]=v[331];
    out[511]=v[332];
    out[512]=v[333];
    out[513]=v[334];
    out[514]=v[335];
    out[515]=v[336];
    out[522]=v[337];
    out[524]=1.0;
    out[680]=v[338];
    out[681]=v[339];
    out[682]=v[340];
    out[683]=v[341];
    out[776]=v[342];
    out[777]=v[343];
    out[778]=v[344];
    out[779]=v[345];
    out[812]=v[346];
    out[528]=v[347];
    out[529]=v[348];
    out[530]=v[349];
    out[531]=v[350];
    out[532]=v[351];
    out[533]=v[352];
    out[534]=v[353];
    out[535]=v[354];
    out[536]=v[355];
    out[537]=v[356];
    out[538]=v[357];
    out[539]=v[358];
    out[546]=v[359];
    out[549]=v[360];
    out[684]=v[361];
    out[685]=v[362];
    out[686]=v[363];
    out[687]=v[364];
    out[780]=v[365];
    out[781]=v[366];
    out[782]=v[367];
    out[783]=v[368];
    out[813]=v[369];
    out[552]=v[370];
    out[553]=v[371];
    out[554]=v[372];
    out[555]=v[373];
    out[556]=v[374];
    out[557]=v[375];
    out[558]=v[376];
    out[559]=v[377];
    out[560]=v[378];
    out[561]=v[379];
    out[562]=v[380];
    out[563]=v[381];
    out[570]=v[382];
    out[574]=1.0;
    out[688]=v[383];
    out[689]=v[384];
    out[690]=v[385];
    out[691]=v[386];
    out[784]=v[387];
    out[785]=v[388];
    out[786]=v[389];
    out[787]=v[390];
    out[814]=v[391];
    out[578]=v[392];
    out[579]=v[393];
    out[580]=v[394];
    out[581]=v[395];
    out[582]=v[396];
    out[583]=v[397];
    out[584]=v[398];
    out[585]=v[399];
    out[586]=v[400];
    out[587]=v[401];
    out[599]=1.0;
    out[692]=v[402];
    out[693]=v[403];
    out[694]=v[404];
    out[695]=v[405];
    out[788]=v[406];
    out[789]=v[407];
    out[790]=v[408];
    out[791]=v[409];
    out[815]=v[410];
    for(int i=0;i<NX;++i) {
        double affine=v[i];
        for(int j=0;j<NX;++j) affine-=out[NX+i*NX+j]*x0[j];
        for(int j=0;j<NU;++j) affine-=out[NX+NX*NX+i*NU+j]*u0[j];
        for(int j=0;j<NU;++j) affine-=out[NX+NX*NX+NX*NU+i*NU+j]*u1[j];
        if(free_time) affine-=out[NX+NX*NX+2*NX*NU+i]*scale;
        out[OUT-NX+i]=affine;
    }
}
