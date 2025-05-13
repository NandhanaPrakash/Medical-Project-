import 'package:flutter/material.dart';

void main() {
  runApp(Modulo8CounterApp());
}

class Modulo8CounterApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Modulo-10 Counter',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: Modulo8CounterScreen(),
    );
  }
}

class Modulo8CounterScreen extends StatefulWidget {
  @override
  _Modulo8CounterScreenState createState() => _Modulo8CounterScreenState();
}

class _Modulo8CounterScreenState extends State<Modulo8CounterScreen> {
  int _counter = 0;

  void _incrementCounter() {
    setState(() {
      _counter = (_counter + 1) % 10; // Increment and wrap around after 7
    });
  }

  void _resetCounter() {
    setState(() {
      _counter = 0; // Reset to 0
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Modulo-10 Counter'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text(
              'Current Count:',
              style: TextStyle(fontSize: 24),
            ),
            Text(
              '$_counter',
              style: TextStyle(fontSize: 48, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _incrementCounter,
              child: Text('Increment'),
            ),
            SizedBox(height: 10),
            ElevatedButton(
              onPressed: _resetCounter,
              child: Text('Reset'),
            ),
          ],
        ),
      ),
    );
  }
}
