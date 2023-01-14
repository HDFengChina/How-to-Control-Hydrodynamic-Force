import java.util.*;
import org.apache.xmlrpc.*;
import java.util.concurrent.*;
import java.util.concurrent.Semaphore;
import com.alibaba.fastjson.JSONObject;
Pinball test;  
//RotatingFoil test;
SaveScalar dat;
SaveData veldata1;
SaveData veldata2;
SaveData veldata3;
SaveVectorField data;
float maxT;
PrintWriter output1;

int Re = 100;
//String path = "C:\\Users\\haodong.feng\\Documents\\west_puyi\\RLFluidControl\\RLFluidControl\\clientCFD\\saved\\";
float chord = 1.0;
//float dAoA = 25*PI/180, uAoA = dAoA;
int resolution = 16, xLengths = 16, yLengths = 8, zoom = 100/resolution;
int plotTime = 28;
int picNum = 20;
int simNum = 1;
float tStep = .0075;
String datapath = "saved" + "/";
int EpisodeNum = 1;
int initTime = 1, callLearn = 16, act_num = 0;
int EpisodeTime = 30, ActionTime = 0; //TODO: actionTime = 5
float CT = 0, CL = 0, CP = 0, forceX = 0;
float AvgCT = 0, Avgeta = 0;
//float initTime = 0.5;  //In fact, the initial time is 1 in simulation
float AoA = 0, A = 0;
float y = 0, angle = 0;
float Cd = 0, Cl = 0;
float cost = 0, ref = 5, ave_reward=0; //
String p;
XmlRpcClient client;
WebServer server;
ArrayList<PVector> velvalue = new ArrayList<PVector>();
ArrayList<Float> CV15 = new ArrayList<Float>();
ArrayList<Float> CV30 = new ArrayList<Float>();
ArrayList<Float> CV50 = new ArrayList<Float>();
int velcount = 0;
float[] ycoords;

int StepTime = 10, StepCount = 1;// used for adjust Ct avg length

float nextactionC=0, nextactionB=0,nextactionA=0,nextactionAvel = 0, nextactionAoAvel = 0;
float formerA = 0, formerAoA = 0,formerAvel = 0, formerAoAvel = 0;
float[] NextAction = {nextactionA,nextactionB,nextactionC};
float[] FormalAction = {formerA,formerAvel,formerAoA, formerAoAvel};


//remote info exchange
ArrayList state = new ArrayList();
float[] action = new float[3];
float reward_buffer = 0;
Boolean done = false;
int reset_flag = 0;

// Semaphore to provide asynchronization
Semaphore action_sem = new Semaphore(0);
Semaphore state_sem = new Semaphore(0);

void settings()
{
  size(512, 256);

}

void setup()
{

  try
  {
    server = StartServer(int(args[0])); 
  }
  
  catch (Exception ex)
  {
    println(ex);
  }
  setUpNewSim(EpisodeNum);
}

void draw(){
  if (test.t < EpisodeTime){
    //println(test.t);
    // TODO: check reset and done [notice: reset_flag will be changed by server]
    if (test.t>initTime){
      NextAction = callActionByRemote();
      println(Cd);
      println(NextAction);
        //System.out.println("[debug]after action...");
      nextactionA = NextAction[0];
        //nextactionAvel = NextAction[0];
      nextactionB = NextAction[1];
        //nextactionAoAvel = NextAction[1];
      nextactionC = NextAction[2];
      act_num += 1;
      test.xi0 = nextactionA;
      test.xi1 = nextactionB;
      test.xi2 = nextactionC;
      test.xi0_m = 5*test.xi0;
      test.xi1_m = 5*test.xi1;
      test.xi2_m = 5*test.xi2;  //left is positive direction
      callLearn--;
      
      if(test.t>plotTime){
        picNum--;
        
        //output1 = createWriter("saved/field/1650/"+str(test.t)+".txt");  // read flow
        //ArrayList<PVector> aimcoords = test.body.coords;  // read flow
        //PVector coordinate = test.body.xc;  // read flow
        //float py = coordinate.y+64;  // read flow
        //float px = coordinate.x-50;  // read flow
        //addVelField(test.flow.p, test.flow.u, px, py, aimcoords,coordinate);  // read flow
        
        if(picNum <= 0){
          test.display();
          saveFrame("saved/"+str(simNum) + "/" +"frame-#######.png");
          picNum = 10;
        }
      }
    }
    test.update2();
    dat.addData03(test.t, test.xi0, test.xi1, test.xi2, test.force, test.force_1, test.force_2, test.flow.p, test.flow.u);
    String state_cn = multy_state(Cl, Cd);

    //Cd = (test.force.x + test.force_1.x + test.force_2.x)/3; // change average Cd,Cl to the front one in 11_22
    //Cl = (test.force.y + test.force_1.y + test.force_2.y)/3;
    Cd = test.force.x;
    //Cl = test.force.y;
    //Cd = test.force_1.x; // use the Cd Cl of top cylinder to try
    //Cl = test.force_1.y;
    Cl = test.force_2.y;
    Cd = Cd*2/resolution;
    Cl = Cl*2/resolution;
    state_cn = multy_state(Cl, Cd);
    state.clear();
    state.add(state_cn);
    reward(cost);
  }
  else { 
    test.display();
    //saveFrame(path + "\\" + str(simNum) + "\\result.png");        
    dat.finish(); 
    println("Episode:" + simNum);
    simNum = simNum + 1; 
    ave_reward = ave_reward/act_num; //
    //println(act_num); //
    //println(ave_reward); //
    if (ave_reward > -0.1) {exit();} //
    //if (test.t > 30) {exit();}
    setUpNewSim(simNum);
  }        
}

// remote call action
float[] callActionByRemote()
{
  try
  {
    // action_sem will wait, utill the server receive the action
    action_sem.acquire();
  }
  catch (Exception ex)
  {
    System.out.println(ex);
  }
  //System.out.println("[debug]return refreshed action. action0:"+ String.valueOf(action[0])+ "action1:"+ String.valueOf(action[1]));
  //+ "action2:"+ String.valueOf(action[2])+ "action3:"+ String.valueOf(action[3]));
  return action;
}

void reward(float COST)
{
  float target_reward = COST;
  if ((test.t +0.12) > EpisodeTime){
    done = true;
    dat.finish(); 
    simNum = simNum + 1;
    setUpNewSim(simNum);
  }
 
  println("time:" + test.t);
  //done = false;
  // TODO: !!! update reward , done and state in buffer
  reward_buffer = target_reward;
  //release state semaphore to let server return the resulted state
  state_sem.release();
}

// start a server
WebServer StartServer(int port)
{
  println(port);
  WebServer server = new WebServer(port);
  server.addHandler("connect", new serverHandler());
  server.start();

  System.out.println("Started server successfully.");
  System.out.println("Accepting requests. (Halt program to stop.)");
  return server;
}

// server handler to provide api
public
class serverHandler
{

    public String Step(String actionInJson)
      {
        JSONObject input_object = JSONObject.parseObject(actionInJson);
        JSONObject output_object = new JSONObject();
        
        //refresh action TODO: need to pre-processing before using the raw data
        //TODO: Add vel to action
        action[0] = input_object.getFloat("v1");
        action[1] = input_object.getFloat("v2");
        action[2] = input_object.getFloat("v3");
        
        // release action, and then
        action_sem.release();
        //println("[debug]action:", action[0], "  ", action[1]);//, "  ",action[2], "  ", action[3]);

        // query new state, reward , done and wait
        //TODO: maybe some bug
        //println(state_sem);
        try {
            state_sem.acquire();
        } catch (InterruptedException e) {
            // do something, if you wish
            println(e);
            println("[Error] state do not refresh");
        } finally {
          //  state_sem.release();
        }
        //println(state);
        output_object.put("cd", Cd);
        output_object.put("cl", Cl);
        output_object.put("reward", reward_buffer);
        output_object.put("done", done);

        return output_object.toJSONString();
      }

    public String query_state()
      {
        JSONObject output_object = new JSONObject();
        output_object.put("state", state);
        return output_object.toJSONString();
      }
      
    public String reset(String actionInJson)
      {
        JSONObject input_object = JSONObject.parseObject(actionInJson);
        JSONObject output_object = new JSONObject();
        //reset_flag = 1;
        //test.t = 0; // Very important, which make the lilypad out of end loop
        done = false;
        action[0] = input_object.getFloat("v1");
        action[1] = input_object.getFloat("v2");
        action[2] = input_object.getFloat("v3");
        // release action, and then
        action_sem.release();
        // query new state, reward , done and wait
        //TODO: maybe some bug
        try {
            state_sem.acquire();
        } catch (InterruptedException e) {
            // do something, if you wish
            print(e);
        } finally {
           // state_sem.release();
        }
        output_object.put("state", state);
        output_object.put("reward", reward_buffer);
        output_object.put("done", done);
        println("complete reset");
        return output_object.toJSONString();
      }


    //public String reset()
    //  {
    //    JSONObject input_object = JSONObject.parseObject(actionInJson);
    //    JSONObject output_object = new JSONObject();
    //    //reset_flag = 1;
    //    test.t = 0; // Very important, which make the lilypad out of end loop
    //    done = false;
    //    // release action, and then
    //    action_sem.release();
    //    // query new state, reward , done and wait
    //    //TODO: maybe some bug
    //    try {
    //        state_sem.acquire();
    //    } catch (InterruptedException e) {
    //        // do something, if you wish
    //        print(e);
    //    } finally {
    //       // state_sem.release();
    //    }
    //    output_object.put("state", state);
    //    output_object.put("reward", reward_buffer);
    //    output_object.put("done", done);
    //    return "success";
    //  }
}

public String multy_state(float Cl, float Cd) {
  JSONObject multy_state_json = new JSONObject();
  multy_state_json.put("lift", Cl);
  multy_state_json.put("drag", Cd);
 return multy_state_json.toJSONString();
}

void setUpNewSim(int runNum){       
  int xLengths = 16, yLengths = 8, zoom = 100/resolution, Re = 100;//default ufree=1       
  float gR = 0.5;       
  //float act = random(-1,1);
  float xi0 = 0, xi1 = 0, xi2 = 0, theta = PI/6;  
  //float xi0 = 0.2, xi1 = 0.2, xi2 = 0.2, theta = PI/6; 
  //float xi0 = act, xi1 = act, xi2 = act, theta = PI/6;  
  
  act_num = 0; //
  ave_reward = 0; //
  
  smooth();
  
  if (zoom <= 1){zoom = 1;}
  
  test = new Pinball(resolution, Re, gR, theta, xi0, xi1, xi2, tStep, xLengths, yLengths, true);          
  dat = new SaveScalar("saved/"+str(runNum)+".txt", (float)resolution, (float)xLengths, (float)yLengths, 32);      
  
  new File(datapath + str(runNum)).mkdir();
  nextactionA=0;
  nextactionB=0;
  nextactionC=0;
  
  println("Episode");
}

void addVelField(Field p, VectorField u, float px, float py, ArrayList<PVector> aimcoords,PVector coordinate) {  // this function is to read flow
  
    output1.println("%xccoord = ");
    output1.print(coordinate.x +" "+coordinate.y);
    output1.println(";");
    output1.println("% xcoord = ");
    for (int j=0; j<aimcoords.size(); j++) {
      output1.print(aimcoords.get(j).x +" ");
    }
    output1.println(";");

    output1.println("% ycoord = ");
    for (int x=0; x<aimcoords.size(); x++) {
      output1.print(aimcoords.get(x).y +" ");
    }
    output1.println(";");
  //output.println("% pressure = ");
  //for (int i = 0; i < (aimcoords.size()); i++){
  //  output.print(p.linear(aimcoords.get(i).x, aimcoords.get(i ).y) +" ");
  //}
  //output.println(";");

    output1.println("% Vel = ");
    for (int j=round(py); j>=(py-128); j--) { //5C
      //7C+10 in x direction 
      for (int i=round(px); i<=(px+242); i++) {
        
        output1.print(u.x.a[i][j] +" ");
        output1.print(u.y.a[i][j] +" ");
      }
      output1.println(";");
    }
    output1.flush(); // Writes the remaining data to the file
    output1.close(); // Finishes the file
  }
