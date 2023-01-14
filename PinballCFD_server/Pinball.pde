class Pinball {
  BDIM flow;
  BodyUnion body;
  boolean QUICK = true, order2 = true;
  int n, m, out, up, resolution,NT=1;
  float dt, t, D, xi0, xi1, xi2,theta;//initial dt=1 only determine BDIM's update u.
  float xi0_m, xi1_m, xi2_m, gR, theta_m, r, dphi0, dphi1, dphi2;
  FloodPlot flood;
  PVector force, force_0, force_1, force_2;

  Pinball (int resolution, int Re,  float gR,  float theta,  float xi0,  float xi1,  float xi2, float dtReal, int xLengths, int yLengths, boolean isResume) {
    n = xLengths*resolution;
    m = yLengths*resolution;
    this.resolution = resolution;
    this.xi0 = xi0;
    this.xi1 = xi1;
    this.xi2 = xi2;
    this.gR = gR;
    xi0_m=5*xi0;
    xi1_m=5*xi1;
    xi2_m=5*xi2;
    this.theta = theta;
    this.dt = dtReal*this.resolution;
    theta_m=theta;

    Window view = new Window(0, 0, n, m); // zoom the display around the body
    D=resolution;

    float r=D+gR*D;
    body =new BodyUnion(new CircleBody(n/4, m/2, D, view),
    new CircleBody(n/4+r*cos(theta), m/2-r*sin(theta), D, view),
    new CircleBody(n/4+r*cos(theta), m/2+r*sin(theta), D, view));
    flow = new BDIM(n,m,dt,body,(float)D/Re,QUICK);
    
    if(isResume){
      flow.resume("saved/init/init.bdim");
    }
    
    flood = new FloodPlot(view);
    flood.range = new Scale(-1, 1);
    flood.setLegend("vorticity"); 
  }


 void update2(){
   //dt = flow.checkCFL();
   flow.dt = dt;
   //dphi0 = (2*xi0*dt)/D; //anglar velocity
   //dphi1 = (2*xi1*dt)/D;
   //dphi2 = (2*xi2*dt)/D;
   println(xi0_m,xi1_m,xi2_m);
   dphi0 = (2*xi0_m*dt)/D; //anglar velocity
   dphi1 = (2*xi1_m*dt)/D;
   dphi2 = (2*xi2_m*dt)/D;
   body.bodyList.get(0).rotate(dphi0);//change index try;
   body.bodyList.get(1).rotate(dphi1);//change index try;
   body.bodyList.get(2).rotate(dphi2);
              
   flow.update(body);
   if (order2) {flow.update2(body);}
   
   //flow.write("saved\\init\\init_1.bdim"); //
   //print("t="+nfs(t,2,2)+";  ");
   t += dt/resolution;  //nonedimension
  
   force = body.bodyList.get(0).pressForce(flow.p).mult(-1);
   //force_0 = body.bodyList.get(0).pressForce(flow.p).mult(-1);  //multply calculation to -1
   force_1 = body.bodyList.get(1).pressForce(flow.p).mult(-1);
   force_2 = body.bodyList.get(2).pressForce(flow.p).mult(-1);
   //force.add(force_1);
   //force.add(force_2);
   //force_x = force_1.x + force_2.x + force_0.x;
   //force_y = force_1.y + force_2.y + force_0.y;
   //print("drag="+nfs(force.x*2/D, 2, 2)+";  "); //the function of nfs is to let 2 places before dot and 2 places after dot
   //println("lift="+nfs(force.y*2/D, 2, 2)+";  ");
   //println("velocity="+nfs(flow.u.x.a[n/4+5][m/2+1],2,2)+nfs(flow.u.y.a[n/4+5][m/2+1],2,2)+";  "); //output velocity
 }

 void update(){
    for ( int i=0 ; i<NT ; i++ ) {
      if (flow.QUICK) {
        dt = flow.checkCFL();
        flow.dt = dt;
      }
       
       dphi0 = (2*xi0*dt)/D;
       dphi1 = (2*xi1*dt)/D;
       dphi2 = (2*xi2*dt)/D;
       body.bodyList.get(0).rotate(dphi0);//change index try;
       body.bodyList.get(1).rotate(dphi1);//change index try;
       body.bodyList.get(2).rotate(dphi2);
              
       flow.update(body);
       if (order2) {flow.update2(body);}
       //print("t="+nfs(t,2,2)+";  ");
       t += dt/resolution;  //nonedimension
  
       force = body.bodyList.get(0).pressForce(flow.p).mult(-1);
       //force_0 = body.bodyList.get(0).pressForce(flow.p).mult(-1);  //multply calculation to -1
       force_1 = body.bodyList.get(1).pressForce(flow.p).mult(-1);
       force_2 = body.bodyList.get(2).pressForce(flow.p).mult(-1);
       //force.add(force_1);
       //force.add(force_2);
       //force_x = force_1.x + force_2.x + force_0.x;
       //force_y = force_1.y + force_2.y + force_0.y;
       //print("drag="+nfs(force.x*2/D, 2, 2)+";  "); //the function of nfs is to let 2 places before dot and 2 places after dot
       //println("lift="+nfs(force.y*2/D, 2, 2)+";  ");
       //println("velocity="+nfs(flow.u.x.a[n/4+5][m/2+1],2,2)+nfs(flow.u.y.a[n/4+5][m/2+1],2,2)+";  "); //output velocity
       //println("aaa" + flow.u.x.linear( 4.65, 4.15 ) + " " + flow.u.y.linear( 4.65, 4.15 )); 
  }
}

  void display() {
    flood.display(flow.u.curl());
    body.display();
    flood.displayTime(t);
  }
}
