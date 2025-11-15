// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StudentData {

    // ---- STRUCT ----
    struct Student {
        uint id;
        string name;
        uint age;
    }

    // ---- ARRAY OF STRUCTS ----
    Student[] public students;

    // ---- ADD STUDENT ----
    function addStudent(uint _id, string memory _name, uint _age) public {
        students.push(Student(_id, _name, _age));
    }

    // ---- GET STUDENT BY INDEX ----
    function getStudent(uint index) public view returns(uint, string memory, uint) {
        require(index < students.length, "Index out of range");
        Student memory s = students[index];
        return (s.id, s.name, s.age);
    }

    // ---- NUMBER OF STUDENTS ----
    function totalStudents() public view returns(uint) {
        return students.length;
    }

    // ---- FALLBACK FUNCTION ----
    // Triggered when calling a non-existing function
    fallback() external payable {
        // Ether sent here will be logged but not used
    }

    // ---- RECEIVE FUNCTION ----
    // Triggered when contract receives plain ETH
    receive() external payable {}
}