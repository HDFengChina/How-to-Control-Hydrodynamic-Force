class RotatingFoil {
  final int n, m;
  float dt = 0, t = 0, dAoA,dA,chord = 1.0, dfrac=0.5;
  float AoF, v2, pitch=0,Heave=0, p=0;
  float F;
  float b=50; //note b/resolution is true time constant...
  int resolution;

  boolean upstroke = false;

  NACA foil; 
  BDIM flow; 
  FloodPlot flood, flood2; 
  Window window;
  ReadData reader;
  PVector force;

  RotatingFoil( int resolution, int xLengths, int yLengths, float dtReal, int Re, boolean QUICK,boolean isR) { // isR for flow init decsion
    this.resolution = resolution;
    n = xLengths*resolution;
    m = yLengths*resolution;
    window = new Window(n, m);

    foil = new NACA(n/2+96, m/2, resolution*chord, .16, window); // NACA0016
    // x position: n/2+100
    foil.rotate(-foil.phi+PI);
    foil.rotate(0);
    
    //this.dt = dtReal*this.resolution;
    flow = new BDIM(n, m, dt, foil, (float)resolution/Re, QUICK, -1); // flow is from right to left, which accords to the right-hand coord
    if(isR){
      flow.resume("./init.bdim");
    }
    flood = new FloodPlot(window);
    flood.range = new Scale(-0.5, 0.5);
    flood.setLegend("vorticity");
    flood.setColorMode(1); 
    foil.setColor(#CCCCCC);
  }
  
  void setFlapParams(float dAoA,float dA) {
    this.dAoA = dAoA; 
    this.dA = dA; 
  }
  
  void motion(float dAoA,float dA){
  foil.rotate(-dAoA);
  foil.translate(0.,dA);
  }

  void computeState(float t) {
    AoF = atan2(0., 1.);
    v2 = 1;
    PVector pforce = foil.pressForce(flow.p);
    F = pforce.y*cos(AoF)+pforce.x*sin(AoF);
  }

  void update2(float AoA,float A){
    if (flow.QUICK) {
      dt = flow.checkCFL();
      flow.dt = dt;
    }
    //flow.dt = this.dt;
    //foil.rotate(-AoA/180*PI+PI-foil.phi);
    //foil.translate(0.,A-foil.xc.y);
    foil.rotate(AoA*flow.dt);
    foil.translate(0, A*dt);
    //foil.rotate(-foil.phi-AoA*flow.dt+PI);
    //foil.translate(0.,A*flow.dt-foil.xc.y+m/2.);
    flow.update(foil);
    flow.update2();
    t += dt/resolution;  //nonedimension
    
    force = foil.pressForce(flow.p);
    //[debug]
    //print("t="+nfs(t,2,3)+";  ");
    //print("drag="+nfs(force.x*2/this.resolution, 2, 2)+";  ");
    //print("lift="+nfs(force.y*2/this.resolution, 2, 2)+";  ");
  }
  
  void update() {
    if (flow.QUICK) {
      dt = flow.checkCFL();
      flow.dt = dt;
    }

    computeState(t);
    foil.rotate(-AoA/180*PI+PI-foil.phi);
    foil.translate(0.,A-foil.xc.y);
    //println("AoA: "+(pitch-AoF)*180/PI);

    flow.update(foil);flow.update2();
    t += dt;
    
    print("t="+nfs(t/resolution,2,2)+";  ");
  }
  
  void display() {
    flood.display(flow.u.curl());
    foil.display();
    //foil.displayVector(foil.pressForce(flow.p));
  }
}
