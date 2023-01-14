/***
import java.util.*;
import org.apache.xmlrpc.*;
import java.util.concurrent.*;
import java.util.concurrent.Semaphore;
import com.alibaba.fastjson.JSONObject;
RotatingFoil test;
SaveData dat;
float maxT;
PrintWriter output;

int Re = 1000;
float chord = 1.0;
//float dAoA = 25*PI/180, uAoA = dAoA;
int resolution = 32, xLengths = 7, yLengths = 5, zoom = 3;
int picNum = 10;
float tStep = .005;
String datapath = "saved" + "/";
int EpisodeNum = 1;
int EpisodeTime = 50, ActionTime = -1; //TODO: actionTime = 5
float CT = 0, CL = 0, CP = 0;
float AoA = 0, A = 0;
float y = 0, angle = 0;
String p;
XmlRpcClient client;
WebServer server;
float[] ycoords;
int calllearn = 2;

//remote info exchange
float[] state = new float[6];
float[] action = new float[2];
float reward_buffer = 0;
Boolean done = false;
int reset_flag = 0;
// Semaphore to provide asynchronization
Semaphore action_sem = new Semaphore(0);
Semaphore state_sem = new Semaphore(0);

void settings()
{
  size(zoom * xLengths * resolution, zoom * yLengths * resolution);
  state[0] = 0;
  state[1] = 0;
  state[2] = 0;
  state[3] = 0;
  state[4] = 0;
  state[5] = 0;
}

void setup()
{
  surface.setVisible(false);
  try
  {
    //client = new XmlRpcClient("http://localhost:8060");
    server = StartServer();
    Vector params = new Vector();
    params.addElement(new Integer(-1));
    //Object result = client.execute("init", params);
    //print(result);
  }
  catch (Exception ex)
  {
    println(ex);
  }
  setUpNewSim(EpisodeNum);
}
void draw()
{
  if (test.t <= EpisodeTime)
  {
    // TODO: check reset  and done [notice: reest_flag will be changed by server]
    if(reset_flag == 1){
        // TODO: call reset function
    }
    if(done == true){
        continue
    }
    // TODO:maybe call action first?
    test.update2(AoA, A);
    test.display();
    //dat.addData(test.t, test.foil.pressForce(test.flow.p), test.foil, test.flow.p);
    //dat.addDataSimple(test.t, test.foil.pressForce(test.flow.p));
    PVector forces = test.foil.pressForce(test.flow.p);
    ycoords = ycoords(test);
    angle = (test.foil.phi - PI) / PI * 180.;
    PVector coordinate = test.foil.xc;
    y = coordinate.y - yLengths * resolution / 2.;
    if (test.t > ActionTime)
    {
      calllearn--;
      CT += -forces.x / resolution * 2;
      AoA = test.foil.phi * 180 / PI - 180;
      A = test.foil.xc.y;
      if (calllearn < 0)
      {
        // TODO: state should change !!
        state[0] = state[2];
        state[1] = state[3];
        state[2] = state[4];
        state[3] = state[5];
        state[4] = y;
        state[5] = angle;
        calllearn = 2;
        CT = CT / calllearn;
        String State = String.valueOf(state[0]);
        for (int i = 1; i < state.length; i++)
        {
          State = State + "_" + String.valueOf(state[i]);
        }
        //float[] NextAction = callAction(State); // TODO: read action from server logger
        System.out.println("[debug]before action...");
        float[] NextAction = callActionByRemote(State);
        System.out.println("[debug]after action...");
        AoA = NextAction[0];
        A = NextAction[1];
        if (abs(AoA - angle) > 0.15)
        {
          AoA = test.foil.phi * 180 / PI - 180 + sign(AoA - angle) * 0.15;
        }
        if (abs(A - test.foil.xc.y) > 0.3)
        {
          A = coordinate.y + sign(A - y) * 0.3;
        }
        println("AoA= " + AoA + " " + "A= " + A + " " + "y= " + test.foil.xc.y + " " + "theta= " + test.foil.phi * 180 / PI);
        if ((max(ycoords) >= yLengths * resolution - 5) || (min(ycoords) <= 5))
        {
          reward(-100); // TODO: prepare reward and done infomation
          dat.finish();
          output.close();
          EpisodeNum += 1;
          setUpNewSim(EpisodeNum);
          test.t = 0;
        }
        else
        {
          reward(CT);
        }
        output.println("" + AoA + "," + A + "," + CT);
        CT = 0;
      }
      dat.output.println(test.t + " " + forces.x + " " + forces.y + " " + angle + " " + test.foil.xc.y + ";");
      println("EpisodeNUm=" + EpisodeNum);
      if (test.t > 40)
      {
        picNum--;
        if (picNum <= 0)
        {
          saveFrame(datapath + "Episode" + str(EpisodeNum) + "/" + "frame-#######.png");
          picNum = 10;
        }
      }
    }
  }
  else
  {
    try
    {
      Vector params = new Vector();
      params.addElement(new Integer(1000));
      Object result = client.execute("train", params);
      print(result);
      client.execute("save", params);
      client.execute("save_ANN_weights", params);
    }
    catch (Exception ex)
    {
      println(ex);
    }
    dat.finish();
    output.close();
    EpisodeNum += 1;
    setUpNewSim(EpisodeNum);
    test.t = 0;
  }
}
// call Action
float[] callAction(String state)
{
  float[] XI = new float[2];
  Vector params = new Vector();
  params.addElement(state);

  try
  {
    Object result = client.execute("request_stochastic_action", params);
    String[] newString = ((String)result).split("_");
    XI[0] = 80 * Float.parseFloat(newString[0]);
    XI[1] = 80 + 80 * Float.parseFloat(newString[1]);
  }
  catch (Exception ex)
  {
    println(ex);
  }

  return XI;
}

// remote call action
float[] callActionByRemote(String state)
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
  System.out.println("[debug]return refreshed action. action0:"+ String.valueOf(action[0])+ "action1:"+ String.valueOf(action[1]));
  return action;
}
void reward(float CT)
{
  Vector params = new Vector();
  if (abs(CT) > 100)
  {
    CT = -100;
  }
  params.addElement(String.valueOf(CT));
  // TODO: !!! update reward , done and state in buffer
  reward_buffer = CT;
  //release state semaphore to let server return the resulted state
  state_sem.release();
}
// start a server
WebServer StartServer()
{
  WebServer server = new WebServer(8999);
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
        // sparse json
        JSONObject input_object = JSONObject.parseObject(actionInJson);
        JSONObject output_object = new JSONObject();
        //refresh action TODO: need to pre-processing before using the raw data
        action[0] = input_object.getFloat("y");
        action[1] = input_object.getFloat("theta");
        // release action, and then
        action_sem.release();
        print("action:", action[0], "  ", action[1]);

        // query new state, reward , done and wait
        //TODO: maybe some bug
        try {
            state_sem.acquire();
        } catch (InterruptedException e) {
            // do something, if you wish
            print(e);
        } finally {
            state_sem.release();
        }
        output_object.put("state", state);
        output_object.put("reward", reward_buffer);
        output_object.put("done", false);


        return output_object.toJSONString() ;
      }

    public String query_state()
      {
        JSONObject output_object = new JSONObject();
        output_object.put("state", state);
        return output_object.toJSONString();
      }

    public String reset()
      {
        reset_flag = 1;
        return "success";
      }

}

void
setUpNewSim(int runNum)
{
  new File(datapath + "Episode" + str(runNum)).mkdir();
  test = new RotatingFoil(resolution, xLengths, yLengths, tStep, Re, true);
  dat = new SaveData(datapath + "Episode" + str(runNum) + "/force.txt", test.foil.coords, resolution, xLengths, yLengths, zoom);
  dat.output.println("t" + " " + "fx" + " " + "fy" + " " + "theta" + " " + "y");
  output = createWriter(datapath + "Episode" + str(runNum) + "/output.csv");
  output.println("" + "Action1" + "," + "Action2" + "," + "CT");
  try
  {
    Vector params = new Vector();
    params.addElement(new Integer(-1));
    //client.execute("start_episode", params);
  }
  catch (Exception ex)
  {
    println(ex);
  }
  AoA = test.foil.phi * 180 / PI - 180;
  A = test.foil.xc.y;
}
//SparsePressure
String SparsePressure(RotatingFoil test)
{
  float[] Pressure = new float[0];
  for (int i = 0; i < (test.foil.coords.size()) / 20; i++)
  {
    Pressure = append(Pressure, test.flow.p.linear(test.foil.coords.get(i * 20).x, test.foil.coords.get(i * 20).y));
  }
  Pressure = append(Pressure, test.flow.p.linear(test.foil.coords.get(199).x, test.foil.coords.get(199).y));
  p = String.valueOf(Pressure[0]);
  for (int i = 1; i < Pressure.length; i++)
  {
    p = p + "_" + String.valueOf(Pressure[i]);
  }
  return p;
}
float[] ycoords(RotatingFoil test)
{
  float[] ycoords = new float[0];
  for (int i = 0; i < (test.foil.coords.size()); i++)
  {
    ycoords = append(ycoords, test.foil.coords.get(i).y);
  }
  return ycoords;
}

int sign(float x)
{
  int s = 0;
  if (x > 0)
  {
    s = 1;
  }
  if (x < 0)
  {
    s = -1;
  }
  if (x == 0)
  {
    s = 0;
  }
  return s;
}
***/
